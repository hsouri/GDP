"""Main class, holding information about models and training/testing routines."""

import torch
import warnings
import torchvision
import os

from torchvision import transforms
from ..utils import cw_loss
from ..consts import NON_BLOCKING, BENCHMARK
torch.backends.cudnn.benchmark = BENCHMARK
import numpy as np
from tqdm import tqdm

from ..victims.victim_single import _VictimSingle
from ..victims.batched_attacks import construct_attack
from ..victims.training import _split_data

from .utils import OptimizerDetails
from ..pytorch_diffusion import Diffusion

class _Witch():
    """Brew poison with given arguments.

    Base class.

    This class implements _brew(), which is the main loop for iterative poisoning.
    New iterative poisoning methods overwrite the _define_objective method.

    Noniterative poison methods overwrite the _brew() method itself.

    “Double, double toil and trouble;
    Fire burn, and cauldron bubble....

    Round about the cauldron go;
    In the poison'd entrails throw.”

    """

    def __init__(self, args, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize a model with given specs..."""
        self.args, self.setup = args, setup
        self.retain = True if self.args.ensemble > 1 and self.args.local_rank is None else False
        self.stat_optimal_loss = None

    """ BREWING RECIPES """

    def brew(self, victim, kettle):
        """Recipe interface."""
        if len(kettle.poisonset) > 0:
            if len(kettle.targetset) > 0:
                if self.args.eps > 0:
                    if self.args.budget > 0:
                        poison_delta = self._brew(victim, kettle)
                    else:
                        poison_delta = kettle.initialize_poison(initializer='zero')
                        warnings.warn('No poison budget given. Nothing can be poisoned.')
                else:
                    poison_delta = kettle.initialize_poison(initializer='zero')
                    warnings.warn('Perturbation interval is empty. Nothing can be poisoned.')
            else:
                poison_delta = kettle.initialize_poison(initializer='zero')
                warnings.warn('Target set is empty. Nothing can be poisoned.')
        else:
            poison_delta = kettle.initialize_poison(initializer='zero')
            warnings.warn('Poison set is empty. Nothing can be poisoned.')

        return poison_delta

    def _brew(self, victim, kettle):
        """Run generalized iterative routine."""
        print('Starting brewing procedure ...')
        self._initialize_brew(victim, kettle)
        poisons, scores = [], torch.ones(self.args.restarts) * 10_000

        for trial in range(self.args.restarts):
            poison_delta, target_losses = self._run_trial(victim, kettle)
            scores[trial] = target_losses
            poisons.append(poison_delta.detach())
            if self.args.dryrun:
                break

        optimal_score = torch.argmin(scores)
        self.stat_optimal_loss = scores[optimal_score].item()
        print(f'Poisons with minimal target loss {self.stat_optimal_loss:6.4e} selected.')
        poison_delta = poisons[optimal_score]

        return poison_delta


    def _initialize_brew(self, victim, kettle):
        """Implement common initialization operations for brewing."""
        victim.eval(dropout=True)
        # Compute target gradients
        self.targets = torch.stack([data[0] for data in kettle.targetset], dim=0).to(**self.setup)
        self.intended_classes = torch.tensor(kettle.poison_setup['intended_class']).to(device=self.setup['device'], dtype=torch.long)
        self.true_classes = torch.tensor([data[1] for data in kettle.targetset]).to(device=self.setup['device'], dtype=torch.long)


        # Precompute target gradients
        if self.args.target_criterion in ['cw', 'carlini-wagner']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes, cw_loss)
        elif self.args.target_criterion in ['untargeted-cross-entropy', 'unxent']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.true_classes)
            for grad in self.target_grad:
                grad *= -1
        elif self.args.target_criterion in ['xent', 'cross-entropy']:
            self.target_grad, self.target_gnorm = victim.gradient(self.targets, self.intended_classes)
        else:
            raise ValueError('Invalid target criterion chosen ...')
        print(f'Target Grad Norm is {self.target_gnorm}')

        if self.args.repel != 0:
            self.target_clean_grad, _ = victim.gradient(self.targets, self.true_classes)
        else:
            self.target_clean_grad = None

        # The PGD tau that will actually be used:
        # This is not super-relevant for the adam variants
        # but the PGD variants are especially sensitive
        # E.G: 92% for PGD with rule 1 and 20% for rule 2
        if self.args.attackoptim in ['PGD', 'GD']:
            # Rule 1
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
        elif self.args.attackoptim in ['momSGD', 'momPGD']:
            # Rule 1a
            self.tau0 = self.args.eps / 255 / kettle.ds * self.args.tau * (self.args.pbatch / 512) / self.args.ensemble
            self.tau0 = self.tau0.mean()
        else:
            # Rule 2
            self.tau0 = self.args.tau * (self.args.pbatch / 512) / self.args.ensemble

        # Prepare adversarial attacker if necessary:
        if self.args.padversarial is not None:
            if not isinstance(victim, _VictimSingle):
                raise ValueError('Test variant only implemented for single victims atm...')

            attack = dict(type=self.args.padversarial, strength=self.args.defense_strength)
            if 'overestimate' in self.args.padversarial:
                attack['strength'] *= 2
            self.attacker = construct_attack(attack, victim.model, victim.loss_fn, kettle.dm, kettle.ds,
                                             tau=kettle.args.tau, init='randn', optim='signAdam',
                                             num_classes=len(kettle.trainset.classes), setup=kettle.setup)

        # Prepare adaptive mixing to dilute with additional clean data
        if self.args.pmix:
            self.extra_data = iter(kettle.trainloader)


    def _run_trial(self, victim, kettle):
        """Run a single trial."""
        poison_delta = kettle.initialize_poison()
        if self.args.full_data:
            dataloader = kettle.trainloader
        else:
            dataloader = kettle.poisonloader

        if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
            # poison_delta.requires_grad_()
            if self.args.attackoptim in ['Adam', 'signAdam']:
                att_optimizer = torch.optim.Adam([poison_delta], lr=self.tau0, weight_decay=0)
            else:
                att_optimizer = torch.optim.SGD([poison_delta], lr=self.tau0, momentum=0.9, weight_decay=0)
            if self.args.scheduling:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(att_optimizer, milestones=[self.args.attackiter // 2.667, self.args.attackiter // 1.6,
                                                                                            self.args.attackiter // 1.142], gamma=0.1)
            poison_delta.grad = torch.zeros_like(poison_delta)
            dm, ds = kettle.dm.to(device=torch.device('cpu')), kettle.ds.to(device=torch.device('cpu'))
            poison_bounds = torch.zeros_like(poison_delta)
        else:
            poison_bounds = None

        base_poisons_dict = {}
        prev_inputs = None


        for step in range(self.args.attackiter):
            target_losses = 0
            poison_correct = 0
            for batch, example in enumerate(dataloader):
                
                ###### GDP 
                if self.args.diffusion_base_poisons:
                    if batch not in base_poisons_dict:
                        if self.args.load_poisons is not None:
                            print('Loading GDP base poisons ...')
                            f_index = batch * self.args.pbatch
                            l_index = f_index + example[0].shape[0]
                            base_poisons_dict[batch] = torch.load(self.args.load_poisons)[f_index:l_index]
                            operation = self._get_base_poisons(None, None, victim, kettle, return_operation=True)
                        else:
                            print('Generating GDP base poisons ...')
                            base_poisons = []
                            prev_inputs = example[0]
                            for i_ in tqdm(range(example[0].shape[0]), desc='Base Poison Generation'):
                                base_poison, operation = self._get_base_poisons(example[0][i_:i_+1], example[1][i_:i_+1], victim, kettle)
                                base_poisons.append(base_poison)
                            base_poisons = torch.cat(base_poisons, dim=0)
                            base_poisons_dict[batch] = base_poisons.detach().to(device=torch.device('cpu'))
                    if not self.args.nonormalize:
                        if 'ImageNet' in self.args.dataset:
                            example[0] = (operation.imagenet_transform(base_poisons_dict[batch]) - dm) / ds
                        else:
                            example[0] = (base_poisons_dict[batch] - dm) / ds
                    else:
                        example[0] = base_poisons_dict[batch]
                #######

                loss, prediction = self._batched_step(poison_delta, poison_bounds, example, victim, kettle)
                target_losses += loss
                poison_correct += prediction

                if self.args.dryrun:
                    break

            # Note that these steps are handled batch-wise for PGD in _batched_step
            # For the momentum optimizers, we only accumulate gradients for all poisons
            # and then use optimizer.step() for the update. This is math. equivalent
            # and makes it easier to let pytorch track momentum.
            if self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                if self.args.attackoptim in ['momPGD', 'signAdam']:
                    poison_delta.grad.sign_()
                att_optimizer.step()
                if self.args.scheduling:
                    scheduler.step()
                att_optimizer.zero_grad()
                with torch.no_grad():
                    # Projection Step
                    poison_delta.data = torch.max(torch.min(poison_delta, self.args.eps /
                                                            ds / 255), -self.args.eps / ds / 255)
                    poison_delta.data = torch.max(torch.min(poison_delta, (1 - dm) / ds -
                                                            poison_bounds), -dm / ds - poison_bounds)

            target_losses = target_losses / (batch + 1)
            poison_acc = poison_correct / len(dataloader.dataset)
            if step % (self.args.attackiter // 5) == 0 or step == (self.args.attackiter - 1):
                print(f'Iteration {step}: Target loss is {target_losses:2.4f}, '
                      f'Poison clean acc is {poison_acc * 100:2.2f}%')

            if self.args.step:
                if self.args.clean_grad:
                    victim.step(kettle, None, self.targets, self.true_classes)
                else:
                    victim.step(kettle, poison_delta, self.targets, self.true_classes)

            if self.args.dryrun:
                break
    
        if self.args.diffusion_base_poisons:
            num_poisons = 0
            matching_scores = {}
            for batch, example in enumerate(dataloader):
                poison_slices, batch_positions = [], []
                prev_inputs, labels, ids = example
                base_poisons = base_poisons_dict[batch].clone().detach().to(device=torch.device('cpu'))
                if not self.args.nonormalize:
                    if 'ImageNet' in self.args.dataset:
                        base_poisons = (operation.imagenet_transform(base_poisons) - dm) / ds
                    else:
                        base_poisons = (base_poisons - dm) / ds
                for batch_id, image_id in enumerate(ids.tolist()):
                    lookup = kettle.poison_lookup.get(image_id)
                    if lookup is not None:
                        poison_slices.append(lookup)
                        batch_positions.append(batch_id)

                if batch_positions:
                    poison_delta[poison_slices] = poison_delta[poison_slices].detach().to(device=torch.device('cpu'))
                    poison_delta[poison_slices] += base_poisons[batch_positions].detach().to(device=torch.device('cpu')) 
                    poison_delta[poison_slices] -= prev_inputs[batch_positions].detach().to(device=torch.device('cpu'))
                    
                    if self.args.filter_max_matching:
                        print('Filtering poisons ...')
                        for p_index, b_index in zip(poison_slices, batch_positions):
                            sample_ = poison_delta[p_index].clone().detach().to(device=torch.device('cpu'))
                            sample_ += prev_inputs[b_index].clone().detach().to(device=torch.device('cpu'))
                            label_ = labels[b_index].clone().to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                            if self.args.ensemble == 1:
                                matching_loss, poison_norm = self._compute_gmloss(victim, torch.unsqueeze(sample_,0), torch.unsqueeze(label_,0), kettle)
                            else:
                                matching_loss, poison_norm = self._compute_gmloss_ensemble(victim, torch.unsqueeze(sample_,0), torch.unsqueeze(label_,0), kettle)
                            matching_scores[p_index] = matching_loss
                    else:
                        num_poisons += base_poisons[batch_positions].shape[0]

            if self.args.filter_max_matching:
                sorted_matching = dict(sorted(matching_scores.items(), key=lambda item: item[1]))
                sorted_matching_list = list(sorted_matching.keys())
                num_poisons = self.args.num_poisons
                poison_delta[sorted_matching_list[num_poisons:]] = 0.0

            self.args.num_poisons = num_poisons
        return poison_delta, target_losses
    

    def _diff_passenger_loss(self, poison_grad, target_grad, target_clean_grad, target_gnorm):
        """Compute the blind passenger loss term."""
        passenger_loss = 0
        poison_norm = 0
        target_norm_ = 0

        indices = torch.arange(len(target_grad))

        for i in indices:
            passenger_loss -= (target_grad[i] * poison_grad[i]).sum()
            poison_norm += poison_grad[i].pow(2).sum()
            if type(target_grad[i]) is float:
                target_norm_ += target_grad[i] * target_grad[i]
            else:
                target_norm_ += target_grad[i].detach().pow(2).sum()
        poison_norm = poison_norm.sqrt()
        target_norm_ = target_norm_.sqrt()

        passenger_loss = passenger_loss / target_norm_  # this is a constant
        # passenger_loss = passenger_loss / target_gnorm
        passenger_loss = 1 + passenger_loss / poison_norm

        return passenger_loss, poison_norm
    

    def _diff_define_objective(self, inputs, labels, criterion, targets, intended_classes, true_classes):
            """Implement the closure here."""
            def closure(model, optimizer, target_grad, target_clean_grad, target_gnorm):
                """This function will be evaluated on all GPUs."""  # noqa: D401
                outputs = model(inputs)
                poison_loss = criterion(outputs, labels)
                prediction = (outputs.data.argmax(dim=1) == labels).sum()
                poison_grad = torch.autograd.grad(poison_loss, model.parameters(), retain_graph=True, create_graph=True)

                passenger_loss, poison_norm = self._diff_passenger_loss(poison_grad, target_grad, target_clean_grad, target_gnorm)

                # passenger_loss.backward(retain_graph=self.retain)
                return passenger_loss, prediction.detach().cpu(), poison_norm, poison_loss
            return closure


    def _forward_loss(self, victim, poisons, labels, operation, get_stats=False, normalize=True):

        if operation.normalize and normalize:
            poisons = (poisons - operation.dm) / operation.ds
        if self.args.paugment and self.args.bpaugment:
            poisons = operation.augment(poisons)

        criterion = torch.nn.CrossEntropyLoss()

        closure = self._diff_define_objective(poisons, labels, criterion, self.targets, self.intended_classes,
                                             self.true_classes)
        
        loss, prediction, poison_norm, poison_loss = victim.compute_diff(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)
        
        forward_loss = operation.matching_loss_w * loss
        
        if operation.poison_loss_w != 0:
            forward_loss += operation.poison_loss_w * poison_loss

    
        matching_loss_value = loss.detach().cpu().item()

        if get_stats:
            return matching_loss_value, poison_norm.detach().cpu().item()

        return forward_loss, matching_loss_value, poison_norm.detach().cpu().item()
    

    def _forwad_loss_base_poisons(self, victim, poisons, labels, operation, get_stats=False, normalize=True):

        if operation.normalize and normalize:
            poisons = (poisons - operation.dm) / operation.ds
        if self.args.paugment and self.args.bpaugment:
            poisons = operation.augment(poisons)
        
        criterion = torch.nn.CrossEntropyLoss()

        outputs = victim.model(poisons)
        poison_loss = criterion(outputs, labels)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        poison_grad = torch.autograd.grad(poison_loss, victim.model.parameters(), retain_graph=True, create_graph=True)

        poison_norm = 0
        indices = torch.arange(len(poison_grad))
        for i in indices:
            poison_norm += poison_grad[i].pow(2).sum()
        poison_norm = poison_norm.sqrt()

        forward_loss = 1.0 / poison_norm

        if operation.poison_loss_w != 0:
            forward_loss += operation.poison_loss_w * poison_loss

        if get_stats:
            return '', poison_norm.detach().cpu().item(), ''
        
        return forward_loss, '', '', poison_norm.detach().cpu().item()
    
    def _get_base_poisons(self, inputs, labels, victim, kettle, return_operation=False):
        if self.args.base_diffusion_model == 'regular':
            base_diffusion = Diffusion.from_pretrained("ema_cifar10", device=self.setup['device'])
        else:
            base_diffusion = None
        # Operation
        operation = OptimizerDetails()

        operation.operation_func = victim
        operation.forward_loss_func = self._forward_loss

        operation.guidance_3 = True
        operation.num_steps = self.args.num_steps
        operation.optim_guidance_3_wt = self.args.optim_guidance_3_wt
        operation.epsilon_w = 1
        operation.guidance_increase_factor = 1
        operation.eps = self.args.eps
        operation.debug = self.args.debug
        operation.print_diff_stats = self.args.print_diff_stats

 
        ### backward guidance is False as we don't use it for GDP
        operation.guidance_2 = False
        operation.poison_loss_w = self.args.poison_loss_w
        operation.matching_loss_w = self.args.matching_loss_w

        # updates for base poison
        operation.num_steps = self.args.base_num_steps
        operation.optim_guidance_3_wt = self.args.base_optim_guidance_3_wt
        operation.poison_loss_w = self.args.base_poison_loss_w
        operation.matching_loss_w = self.args.base_matching_loss_w
        if self.args.base_matching_loss_w == 0.0:
            operation.forward_loss_func = self._forwad_loss_base_poisons

        operation.normalize = not self.args.nonormalize
        if operation.normalize:
            operation.dm = kettle.dm
            operation.ds = kettle.ds
        if self.args.paugment:
            operation.augment = kettle.augment

        if 'ImageNet' in self.args.dataset:
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224)])
            operation.imagenet_transform = transform_train

        if return_operation:
            return operation

        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        inputs = inputs.to(**kettle.setup)

        torch.set_grad_enabled(False)
        base_poisons = base_diffusion.sample_ugd_matching(labels, operation)
        torch.set_grad_enabled(True)

        return base_poisons, operation


    def _compute_gmloss_ensemble(self, victim, inputs, labels, kettle):

        criterion = torch.nn.CrossEntropyLoss()
        
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        
        if self.args.paugment:
            inputs = kettle.augment(inputs)

        closure = self._diff_define_objective(inputs, labels, criterion, self.targets, self.intended_classes,
                                             self.true_classes)
        
        loss, prediction, poison_norm, poison_loss = victim.compute_diff(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)

        return loss.detach().cpu().item(), poison_norm.detach().cpu().item()
    
        
    def _compute_gmloss(self, victim, inputs, labels, kettle):

        criterion = torch.nn.CrossEntropyLoss()
        
        inputs = inputs.to(**kettle.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        
        if self.args.paugment:
            inputs = kettle.augment(inputs)
            
        differentiable_params = [p for p in victim.model.parameters() if p.requires_grad]
        outputs = victim.model(inputs)

        poison_loss = criterion(outputs, labels)
        prediction = (outputs.data.argmax(dim=1) == labels).sum()
        poison_grad = torch.autograd.grad(poison_loss, differentiable_params, retain_graph=True, create_graph=True)

        passenger_loss = self._passenger_loss(poison_grad, self.target_grad, self.target_clean_grad, self.target_gnorm)

        indices = torch.arange(len(poison_grad))
        poison_norm = 0
        for i in indices:
            poison_norm += poison_grad[i].pow(2).sum()
        poison_norm = poison_norm.sqrt()

        return passenger_loss.detach().cpu().item(), poison_norm.detach().cpu().item()

    def _batched_step(self, poison_delta, poison_bounds, example, victim, kettle):
        """Take a step toward minmizing the current target loss."""
        inputs, labels, ids = example

        inputs = inputs.to(**self.setup)
        labels = labels.to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
        # Check adversarial pattern ids
        poison_slices, batch_positions = kettle.lookup_poison_indices(ids)

        # This is a no-op in single network brewing
        # In distributed brewing, this is a synchronization operation
        inputs, labels, poison_slices, batch_positions, randgen = victim.distributed_control(
            inputs, labels, poison_slices, batch_positions)

        # If a poisoned id position is found, the corresponding pattern is added here:
        if len(batch_positions) > 0:
            delta_slice = poison_delta[poison_slices].detach().to(**self.setup)
            if self.args.clean_grad:
                delta_slice = torch.zeros_like(delta_slice)
            delta_slice.requires_grad_()  # TRACKING GRADIENTS FROM HERE
            poison_images = inputs[batch_positions]
            inputs[batch_positions] += delta_slice

            # Add additional clean data if mixing during the attack:
            if self.args.pmix:
                if 'mix' in victim.defs.mixing_method['type']:   # this covers mixup, cutmix 4waymixup, maxup-mixup
                    try:
                        extra_data = next(self.extra_data)
                    except StopIteration:
                        self.extra_data = iter(kettle.trainloader)
                        extra_data = next(self.extra_data)
                    extra_inputs = extra_data[0].to(**self.setup)
                    extra_labels = extra_data[1].to(dtype=torch.long, device=self.setup['device'], non_blocking=NON_BLOCKING)
                    inputs = torch.cat((inputs, extra_inputs), dim=0)
                    labels = torch.cat((labels, extra_labels), dim=0)

            # Perform differentiable data augmentation
            if self.args.paugment:
                inputs = kettle.augment(inputs, randgen=randgen)

            # Perform mixing
            if self.args.pmix:
                inputs, extra_labels, mixing_lmb = kettle.mixer(inputs, labels)

            if self.args.padversarial is not None:
                # The optimal choice of the 3rd and 4th argument here are debatable
                # This is likely the strongest anti-defense:
                if 'v2' in self.args.padversarial:
                    # but the defense itself splits the batch and uses half of it as targets
                    # instead of using the known target [as the defense does not know about the target]
                    delta, additional_info = self.attacker.attack(inputs.detach(), labels, self.true_classes,
                                                                  self.targets, self.intended_classes,
                                                                  steps=victim.defs.novel_defense['steps'])

                else:  # This is a more accurate model of the defense:
                    [temp_targets, inputs,
                     temp_true_labels, labels,
                     temp_fake_label] = _split_data(inputs, labels, target_selection=victim.defs.novel_defense['target_selection'])

                    delta, additional_info = self.attacker.attack(inputs.detach(), labels, temp_true_labels,
                                                                  temp_targets, temp_fake_label, steps=victim.defs.novel_defense['steps'])
                inputs = inputs + delta  # Kind of a reparametrization trick



            # Define the loss objective and compute gradients
            if self.args.target_criterion in ['cw', 'carlini-wagner']:
                loss_fn = cw_loss
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Change loss function to include corrective terms if mixing with correction
            if self.args.pmix:
                def criterion(outputs, labels):
                    loss, pred = kettle.mixer.corrected_loss(outputs, extra_labels, lmb=mixing_lmb, loss_fn=loss_fn)
                    return loss
            else:
                criterion = loss_fn

            closure = self._define_objective(inputs, labels, criterion, self.targets, self.intended_classes,
                                             self.true_classes)
            loss, prediction = victim.compute(closure, self.target_grad, self.target_clean_grad, self.target_gnorm)
            delta_slice = victim.sync_gradients(delta_slice)

            if self.args.clean_grad:
                delta_slice.data = poison_delta[poison_slices].detach().to(**self.setup)

            # Update Step
            if self.args.attackoptim in ['PGD', 'GD']:
                delta_slice = self._pgd_step(delta_slice, poison_images, self.tau0, kettle.dm, kettle.ds)

                # Return slice to CPU:
                poison_delta[poison_slices] = delta_slice.detach().to(device=torch.device('cpu'))
            elif self.args.attackoptim in ['Adam', 'signAdam', 'momSGD', 'momPGD']:
                poison_delta.grad[poison_slices] = delta_slice.grad.detach().to(device=torch.device('cpu'))
                poison_bounds[poison_slices] = poison_images.detach().to(device=torch.device('cpu'))
            else:
                raise NotImplementedError('Unknown attack optimizer.')
        else:
            loss, prediction = torch.tensor(0), torch.tensor(0)

        return loss.item(), prediction.item()

    def _define_objective():
        """Implement the closure here."""
        def closure(model, *args):
            """This function will be evaluated on all GPUs."""  # noqa: D401
            raise NotImplementedError()
            return target_loss.item(), prediction.item()

    def _pgd_step(self, delta_slice, poison_imgs, tau, dm, ds):
        """PGD step."""
        with torch.no_grad():
            # Gradient Step
            if self.args.attackoptim == 'GD':
                delta_slice.data -= delta_slice.grad * tau
            else:
                delta_slice.data -= delta_slice.grad.sign() * tau

            # Projection Step
            delta_slice.data = torch.max(torch.min(delta_slice, self.args.eps /
                                                   ds / 255), -self.args.eps / ds / 255)
            delta_slice.data = torch.max(torch.min(delta_slice, (1 - dm) / ds -
                                                   poison_imgs), -dm / ds - poison_imgs)
        return delta_slice


    def patch_targets(self, kettle):
        """Backdoor trigger attacks need to patch kettle.targets."""
        pass
