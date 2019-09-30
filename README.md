# Model-Free-Episodic-Control
This is the implementation of DQN and Model Free Episodic Control

#Introduction
This package provides DQN and Episodic Control. The DQN implementation is based on [spragunr/deep_q_rl](https://github.com/spragunr/deep_q_rl) and the Episodic Control is written by myself.

[Model Free Episodic Control](http://arxiv.org/abs/1606.04460), C. Blundell et al., *arXiv*, 2016.

[Human-level control through deep reinforcement learning](http://www.nature.com/nature/journal/v518/n7540/pdf/nature14236.pdf), V. Mnih et al., *Nature*, 2015.

I have contacted the author C. Blundell of Model Free Episodic Control. He told me he was using approximate KNN to speed up episodic control, however, he did not tell me details. So I used [annoy](https://github.com/spotify/annoy) to do KNN, and I rebuild the search tree frequently.

related repo:https://github.com/astier/model-free-episodic-control

#Dependencies

Game roms should be stored in directory *roms* which stays next to dqn_ep.

Model-Free-Episodic-Control

├ dqn_ep -> source codes

├ roms -> game roms

└ README.md
		
###Tips: 
I made some changes to DQN so that we do not need OpenCV any more. In addition if your python has OpenAI gym then you do not need to install [Arcade Learning Environment](http://www.arcadelearningenvironment.org/) ([https://github.com/mgbellemare/Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)) Check [spragunr/deep_q_rl](https://github.com/spragunr/deep_q_rl)'s readMe to know more about how to install ALE.

Personally, I recommend using OpenAI gym because it not only can be installed by simply `pip install gym` but also provides us atari game roms (For instance `/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/atari_py/atari_roms` on my mac).




###Dependencies for running DQN

[Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) or [OpenAI gym](https://github.com/openai/gym)
 
 Numpy and SciPy

[Theano](http://deeplearning.net/software/theano/)

[Lasagne](http://lasagne.readthedocs.org/en/latest/)

A reasonable GPU

###Dependencies for running Episodic Control

[Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) or [OpenAI gym](https://github.com/openai/gym)
 
 Numpy and SciPy

[annoy](https://github.com/spotify/annoy) for approximate KNN
 
 A reasonable CPU
 
# Running
examples:

`THEANO_FLAGS='device=gpu0, floatX=float32' python run_nature.py`

`THEANO_FLAGS='device=gpu1, floatX=float32' python run_nature.py -r ms_pacman`

`python run_episodic_control.py`

To get more running details, we can use `python run_episodic_control.py -h` or `python run_nature.py -h`
