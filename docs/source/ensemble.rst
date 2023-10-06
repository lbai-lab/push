.. role:: hidden
    :class: hidden-section

ensemble
===================================

.. automodule:: ensemble
.. currentmodule:: ensemble

Models for Ensembles
-----------------------------

:hidden:`Ensemble`
~~~~~~~~~~~~~~~~~

.. autoclass:: Ensemble
   :members:




.. def mk_optim(params):
..     # Limitiation must be global
..     return torch.optim.Adam(params, lr=1e-5, weight_decay=1e-3)


.. # =============================================================================
.. # Deep Ensemble
.. # =============================================================================

.. def _deep_ensemble_main(particle: Particle, dataloader, loss_fn, epochs) -> None:
..     other_particles = list(filter(lambda x: x != particle.pid, particle.particle_ids()))
..     # Training loop
..     for e in tqdm(range(epochs)):
..         losses = []
..         for data, label in dataloader:
..             loss = particle.step(loss_fn, data, label).wait()
..             losses += [loss]
..             for pid in other_particles:
..                 particle.send(pid, "ENSEMBLE_STEP", loss_fn, data, label)
..         print(f"Average loss {particle.pid}", torch.mean(torch.tensor(losses)))


.. def _ensemble_step(particle: Particle, loss_fn, data, label, *args) -> None:
..     particle.step(loss_fn, data, label, *args)


.. class Ensemble(Infer):
..     def __init__(self, mk_nn: Callable, *args: any, num_devices=1, cache_size=4, view_size=4) -> None:
..         super(Ensemble, self).__init__(mk_nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
        
..     def bayes_infer(self,
..                     dataloader: DataLoader, epochs: int,
..                     loss_fn=torch.nn.MSELoss(),
..                     num_ensembles=2, mk_optim=mk_optim,
..                     ensemble_entry=_deep_ensemble_main, ensemble_state={}, f_save=False):
..         # 1. Create particles
..         pids = [
..             self.push_dist.p_create(mk_optim, device=(0 % self.num_devices), receive={
..                 "ENSEMBLE_MAIN": ensemble_entry
..             }, state=ensemble_state)]
..         for n in range(1, num_ensembles):
..             pids += [self.push_dist.p_create(mk_optim, device=(n % self.num_devices), receive={
..                 "ENSEMBLE_STEP": _ensemble_step,
..             }, state={})]

..         # 2. Perform independent training
..         self.push_dist.p_wait([self.push_dist.p_launch(0, "ENSEMBLE_MAIN", dataloader, loss_fn, epochs)])

..         if f_save:
..             self.push_dist.save()

.. # =============================================================================
.. # Deep Ensemble Training
.. # =============================================================================

.. def train_deep_ensemble(dataloader: Callable, loss_fn: Callable, epochs: int,
..                         nn: Callable, *args, num_devices=1, cache_size=4, view_size=4,
..                         num_ensembles=2, mk_optim=mk_optim,
..                         ensemble_entry=_deep_ensemble_main, ensemble_state={}) -> None:
..     ensemble = Ensemble(nn, *args, num_devices=num_devices, cache_size=cache_size, view_size=view_size)
..     ensemble.bayes_infer(dataloader, epochs, loss_fn=loss_fn, num_ensembles=num_ensembles, mk_optim=mk_optim,
..                          ensemble_entry=ensemble_entry, ensemble_state=ensemble_state)
..     return ensemble.p_parameters()