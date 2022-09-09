#@title import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from colabdesign import mk_afdesign_model, clear_mem
from IPython.display import HTML
import numpy as np

#########################
def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    pass
    # upload_dict = files.upload()
    # pdb_string = upload_dict[list(upload_dict.keys())[0]]
    # with open("tmp.pdb","wb") as out: out.write(pdb_string)
    # return "tmp.pdb"
  else:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"

  ## Fixed backbone design

clear_mem()
DATA_DIR = os.path.join(os.path.dirname(os.getcwd()),'colabdesign','params')
af_model = mk_afdesign_model(protocol="fixbb", data_dir = DATA_DIR)
af_model.prep_inputs(pdb_filename=get_pdb("1TEN"), chain="A")

print("length", af_model._len)
print("weights", af_model.opt["weights"])

af_model.restart()
# default
# af_model.design_3stage()
# shorter
af_model.design_3stage(soft_iters=100, temp_iters=50, hard_iters=5)

af_model.plot_traj()

af_model.save_pdb(f"{af_model.protocol}.pdb")
af_model.plot_pdb()

HTML(af_model.animate())

print(af_model.get_seqs())

  ##Hallucination
# clear_mem()
# af_model = mk_afdesign_model(protocol="hallucination")
# af_model.prep_inputs(length=100)
#
# print("length", af_model._len)
# print("weights", af_model.opt["weights"])
#
# # pre-design with gumbel initialization and softmax activation
# af_model.restart(mode="gumbel")
# af_model.design_soft(50)
#
# # three stage design
# af_model.set_seq(af_model.aux["seq"]["pseudo"])
# af_model.design_3stage(50, 50, 10)
#
# af_model.save_pdb(f"{af_model.protocol}.pdb")
# af_model.plot_pdb()
#
# HTML(af_model.animate())
#
# af_model.get_seqs()
#
#   ##binder hallucination
# clear_mem()
# af_model = mk_afdesign_model(protocol="binder")
# af_model.prep_inputs(pdb_filename=get_pdb("4MZK"), chain="A", binder_len=19)
#
# print("target_length",af_model._target_len)
# print("binder_length",af_model._binder_len)
# print("weights",af_model.opt["weights"])
#
# af_model.restart()
#
# # settings we find work best for helical peptide binder hallucination
# af_model.set_weights(plddt=0.1, pae=0.1, i_pae=1.0, con=0.1, i_con=0.5)
# af_model.set_opt(con=dict(binary=True, cutoff=21.6875, num=af_model._binder_len, seqsep=0))
# af_model.set_opt(i_con=dict(binary=True, cutoff=21.6875, num=af_model._binder_len))
#
# af_model.design_3stage(100,100,10)
#
# af_model.save_pdb(f"{af_model.protocol}.pdb")
# af_model.plot_pdb()
#
# HTML(af_model.animate())
# af_model.get_seqs()
#
#   ##partial hallucination + custom Radius of Gyration (rg) loss
#
# import jax
# import jax.numpy as jnp
# from colabdesign.af.alphafold.common import residue_constants
#
# # first off, let's implement a custom Radius of Gyration loss function
# def rg_loss(inputs, outputs, opt):
#   positions = outputs["structure_module"]["final_atom_positions"]
#   ca = positions[:,residue_constants.atom_order["CA"]]
#   center = ca.mean(0)
#   rg = jnp.sqrt(jnp.square(ca - center).sum(-1).mean() + 1e-8)
#   rg_th = 2.38 * ca.shape[0] ** 0.365
#   rg = jax.nn.elu(rg - rg_th)
#   return {"rg":rg}
#
# clear_mem()
# af_model = mk_afdesign_model(protocol="partial",
#                              loss_callback=rg_loss, # add rg_loss
#                              use_templates=False)   # set True to constrain positions using template input
#
# af_model.opt["weights"]["rg"] = 0.1  # optional: specify weight for rg_loss
#
# af_model.prep_inputs(pdb_filename=get_pdb("6MRR"),
#                      chain="A",
#                      pos="3-30,33-68",  # define positions to contrain
#                      length=100,        # total length if different from input pdb
#                      fix_seq=False)     # set True to constrain sequence in the specified positions
#
# af_model.rewire(loops=[36]) # set loop length between segments
#
# # initialize with wildtype seq, fill in the rest with soft_gumbel distribution
# af_model.restart(mode=["soft","gumbel","wildtype"])
# af_model.design_3stage(100, 100, 10)
#
# af_model.save_pdb(f"{af_model.protocol}.pdb")
# af_model.plot_pdb()
#
# HTML(af_model.animate())
# af_model.get_seqs()
#
