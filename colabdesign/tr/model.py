import random, os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from colabdesign.shared.utils import copy_dict, update_dict, Key, dict_to_str
from colabdesign.shared.prep import prep_pos
from colabdesign.shared.protein import _np_get_6D_binned
from colabdesign.shared.model import design_model, soft_seq

from colabdesign.tr.trrosetta import TrRosetta, get_model_params

# borrow some stuff from AfDesign
from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import protein

try:
  from jax.example_libraries.optimizers import sgd, adam
except:
  from jax.experimental.optimizers import sgd, adam

class mk_tr_model(design_model):
  def __init__(self, protocol="fixbb", num_models=1,
               sample_models=True, data_dir="params/tr",
               loss_callback=None):
    
    assert protocol in ["fixbb","hallucination","partial"]

    self.protocol = protocol
    self._data_dir = "." if os.path.isfile("models/model_xaa.npy") else data_dir
    self._loss_callback = loss_callback
    self._num = 1

    # set default options
    self.opt = {"temp":1.0, "soft":1.0, "hard":1.0, "dropout":False,
                "num_models":num_models,"sample_models":sample_models,
                "weights":{}, "lr":1.0, "bias":0.0, "alpha":1.0, "use_pssm":False}
                
    self._params = {}
    self._inputs = {}

    # setup model
    self._model = self._get_model()
    self._model_params = [get_model_params(f"{self._data_dir}/models/model_xa{k}.npy") for k in list("abcde")]

    if protocol in ["hallucination","partial"]:
      self._bkg_model = TrRosetta(bkg_model=True)
  
  def _get_model(self):
    runner = TrRosetta()    
    def _get_loss(inputs, outputs, opt):
      aux = {"outputs":outputs, "losses":{}}
      log_p = jax.tree_map(jax.nn.log_softmax, outputs)

      # bkg loss
      if self.protocol in ["hallucination","partial"]:
        p = jax.tree_map(jax.nn.softmax, outputs)
        log_q = jax.tree_map(jax.nn.log_softmax, inputs["6D_bkg"])
        aux["losses"]["bkg"] = {}
        for k in ["dist","omega","theta","phi"]:
          aux["losses"]["bkg"][k] = -(p[k]*(log_p[k]-log_q[k])).sum(-1).mean()

      # cce loss
      if self.protocol in ["fixbb","partial"]:
        if self.protocol in ["partial"] and "pos" in opt:
          pos = opt["pos"]
          log_p = jax.tree_map(lambda x:x[:,pos][pos,:], log_p)

        q = inputs["6D"]
        aux["losses"]["cce"] = {}
        for k in ["dist","omega","theta","phi"]:
          aux["losses"]["cce"][k] = -(q[k]*log_p[k]).sum(-1).mean()

      if self._loss_callback is not None:
        aux["losses"].update(self._loss_callback(outputs))

      # weighted loss
      w = opt["weights"]
      tree_multi = lambda x,y: jax.tree_map(lambda a,b:a*b, x,y)
      losses = {k:(tree_multi(v,w[k]) if k in w else v) for k,v in aux["losses"].items()}
      loss = sum(jax.tree_leaves(losses))
      return loss, aux

    def _model(params, model_params, opt, inputs, key):
      seq = soft_seq(params["seq"], opt)
      if "pos" in opt:
        seq_ref = jax.nn.one_hot(inputs["batch"]["aatype"],20)
        p = opt["pos"]
        if self.protocol == "partial":
          fix_seq = lambda x:jnp.where(opt["fix_seq"],x.at[...,p,:].set(seq_ref),x)
        else:
          fix_seq = lambda x:jnp.where(opt["fix_seq"],x.at[...,p,:].set(seq_ref[...,p,:]),x)
        seq = jax.tree_map(fix_seq,seq)
      inputs.update({"seq":seq["pseudo"][0],
                     "prf":jnp.where(opt["use_pssm"],seq["pssm"],seq["pseudo"])[0]})
      rate = jnp.where(opt["dropout"],0.15,0.0)
      outputs = runner(inputs, model_params, key, rate)
      loss, aux = _get_loss(inputs, outputs, opt)
      aux.update({"seq":seq,"opt":opt})
      return loss, aux

    return {"grad_fn":jax.jit(jax.value_and_grad(_model, has_aux=True, argnums=0)),
            "fn":jax.jit(_model)}
  
  def prep_inputs(self, pdb_filename=None, chain=None, length=None,
                  pos=None, fix_seq=True, atoms_to_exclude=None, **kwargs):
    '''
    prep inputs for TrDesign
    '''    
    if self.protocol in ["fixbb", "partial"]:
      # parse PDB file and return features compatible with TrRosetta
      pdb = prep_pdb(pdb_filename, chain, for_alphafold=False)
      self._inputs["batch"] = pdb["batch"]

      if pos is not None:
        self._pos_info = prep_pos(pos, **pdb["idx"])
        p = self._pos_info["pos"]
        if self.protocol == "partial":
          self._inputs["batch"] = jax.tree_map(lambda x:x[p], self._inputs["batch"])

        self.opt["pos"] = p
        self.opt["fix_seq"] = fix_seq

      self._inputs["6D"] = _np_get_6D_binned(self._inputs["batch"]["all_atom_positions"],
                                             self._inputs["batch"]["all_atom_mask"])

      self._len = len(self._inputs["batch"]["aatype"])
      self.opt["weights"]["cce"] = {"dist":1/6,"omega":1/6,"theta":2/6,"phi":2/6}
      if atoms_to_exclude is not None:
        if "N" in atoms_to_exclude:
          # theta = [N]-CA-CB-CB
          self.opt["weights"]["cce"] = dict(dist=1/4,omega=1/4,phi=1/2,theta=0)
        if "CA" in atoms_to_exclude:
          # theta = N-[CA]-CB-CB
          # omega = [CA]-CB-CB-[CA]
          # phi = [CA]-CB-CB
          self.opt["weights"]["cce"] = dict(dist=1,omega=0,phi=0,theta=0)

    if self.protocol in ["hallucination", "partial"]:
      # compute background distribution
      if length is not None: self._len = length
      self._inputs["6D_bkg"] = []
      key = jax.random.PRNGKey(0)
      for n in range(1,6):
        model_params = get_model_params(f"{self._data_dir}/bkgr_models/bkgr0{n}.npy")
        self._inputs["6D_bkg"].append(self._bkg_model(model_params, key, self._len))
      self._inputs["6D_bkg"] = jax.tree_map(lambda *x:jnp.stack(x).mean(0), *self._inputs["6D_bkg"])

      # reweight the background
      self.opt["weights"]["bkg"] = dict(dist=1/6,omega=1/6,phi=2/6,theta=2/6)


    self._opt = copy_dict(self.opt)
    self.restart(**kwargs)

  def set_opt(self, *args, **kwargs):
    '''
    set [opt]ions
    -------------------
    note: model.restart() resets the [opt]ions to their defaults
    use model.set_opt(..., set_defaults=True) 
    or model.restart(..., reset_opt=False) to avoid this
    -------------------    
    model.set_opt(num_models=1)
    model.set_opt(con=dict(num=1)) or set_opt({"con":{"num":1}})
    model.set_opt(lr=1, set_defaults=True)
    '''
    if kwargs.pop("set_defaults", False):
      update_dict(self._opt, *args, **kwargs)

    update_dict(self.opt, *args, **kwargs)

  
  def restart(self, seed=None, opt=None, weights=None,
              seq=None, reset_opt=True, **kwargs):

    if reset_opt:
      self.opt = copy_dict(self._opt)

    self.set_opt(opt)
    self.set_weights(weights)
    self.key = Key(seed=seed).get

    # setup optimizer
    self._init_fun, self._update_fun, self._get_params = sgd(1.0)
    
    # set sequence
    self.set_seq(seq, **kwargs)
    
    self._k = 0

    # clear previous best
    self._best = {}
    
  def run(self, models=None, backprop=True):
    '''run model to get outputs, losses and gradients'''
    
    if models is None:
      # decide which model params to use
      ns = jnp.arange(5)
      m = min(self.opt["num_models"],len(ns))
      if self.opt["sample_models"] and m != len(ns):
        model_num = jax.random.choice(self.key(),ns,(m,),replace=False)
      else:
        model_num = ns[:m]
      model_num = np.array(model_num).tolist()
    elif isinstance(models,int):
      model_num = [models]
    else:
      model_num = list(models)

    # run in serial
    aux_all = []
    for n in model_num:
      model_params = self._model_params[n]
      flags = [self._params, model_params, self.opt, self._inputs, self.key()]
      if backprop:
        (loss,aux),grad = self._model["grad_fn"](*flags)
      else:
        loss,aux = self._model["fn"](*flags)
        grad = jax.tree_map(jnp.zeros_like, self._params)
      aux.update({"loss":loss, "grad":grad})
      aux_all.append(aux)
    
    # average results
    self.aux = jax.tree_map(lambda *x:jnp.stack(x).mean(0), *aux_all)
    self.aux["model_num"] = model_num

  def step(self, models=None, backprop=True,
           callback=None, save_best=True, verbose=1):
    self.run(models=models, backprop=backprop)
    if callback is not None: callback(self)

    # normalize gradient
    g = self.aux["grad"]["seq"]
    gn = jnp.linalg.norm(g,axis=(-1,-2),keepdims=True)

    if "pos" in self.opt and self.opt.get("fix_seq",False):
      # note: gradients only exist in unconstrained positions
      eff_len = self._len - self.opt["pos"].shape[0]
    else:
      eff_len = self._len
    self.aux["grad"]["seq"] *= jnp.sqrt(eff_len)/(gn+1e-7)

    # set learning rate
    self.aux["grad"] = jax.tree_map(lambda x:x*self.opt["lr"], self.aux["grad"])

    # apply gradient
    self._state = self._update_fun(self._k, self.aux["grad"], self._state)
    self._params = self._get_params(self._state)    

    # increment
    self._k += 1

    # save results
    if save_best and "aux" in self._best and self.aux["loss"] < self._best["aux"]["loss"]:
      self._best["aux"] = self.aux

    # print
    if verbose and (self._k % verbose) == 0:
      x = self.get_loss(get_best=False)
      x["models"] = self.aux["model_num"]
      print(dict_to_str(x, print_str=f"{self._k}", keys=["models"]))

  def predict(self, seq=None, models=0):
    self.set_opt(dropout=False)
    if seq is not None:
      self.set_seq(seq=seq, set_state=False)
    self.run(models=models, backprop=False)

  def design(self, iters=100, opt=None, weights=None, save_best=True, verbose=1):
    self.set_opt(opt)
    self.set_weights(weights)
    for _ in range(iters):
      self.step(save_best=save_best, verbose=verbose)
  
  def plot(self, mode="preds", dpi=100, get_best=True):
    '''plot predictions'''

    assert mode in ["preds","feats","bkg_feats"]
    if mode == "preds":
      aux = self._best["aux"] if (get_best and "aux" in self._best) else self.aux
      x = aux["outputs"]
    elif mode == "feats":
      x = self._inputs["6D"]
    elif mode == "bkg_feats":
      x = self._inputs["6D_bkg"]

    x = jax.tree_map(np.asarray, x)

    plt.figure(figsize=(4*4,4), dpi=dpi)
    for n,k in enumerate(["theta","phi","dist","omega"]):
      v = x[k]
      plt.subplot(1,4,n+1)
      plt.title(k)
      plt.imshow(v.argmax(-1),cmap="binary")
    plt.show()
    
  def get_loss(self, k=None, get_best=True):
    aux = self._best["aux"] if (get_best and "aux" in self._best) else self.aux
    if k is None:
      return {k:self.get_loss(k, get_best=get_best) for k in aux["losses"].keys()}
    losses = aux["losses"][k]
    weights = aux["opt"]["weights"][k]
    weighted_losses = jax.tree_map(lambda l,w:l*w, losses, weights)
    return float(sum(jax.tree_leaves(weighted_losses)))
    
  def af_callback(self, weight=1.0, seed=None):
    
    def callback(af_model):      
      # copy [opt]ions from afdesign
      for k,v in af_model.opt.items():
        if k in self.opt and k not in ["weights"]:
          self.opt[k] = af_model.opt[k]

      # update sequence input
      self._params["seq"] = af_model._params["seq"]
      
      # run trdesign
      self.run(backprop = weight > 0)
      
      # add gradients
      af_model.aux["grad"]["seq"] += weight * self.aux["grad"]["seq"]
      
      # add loss
      af_model.aux["loss"] += weight * self.aux["loss"]
        
      # for verbose printout
      if self.protocol in ["hallucination","partial"]:
        af_model.aux["losses"]["TrD_bkg"] = self.get_loss("bkg", get_best=False)
      if self.protocol in ["fixbb","partial"]:
        af_model.aux["losses"]["TrD_cce"] = self.get_loss("cce", get_best=False)
      
    self.restart(seed=seed)
    return callback