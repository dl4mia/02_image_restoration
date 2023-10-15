# DL4MIA 2023: 02 - Image Restoration exercises

**[Return to the Welcome page](https://tinyurl.com/33y2b2hk)**

## Content

---

## Learning Objectives

---

- Supervised training (CARE)
    - Run through the 3D denoising tutorial notebooks
    - Denoise without Ground Truth (Noise2Noise training in CARE)
- Unsupervised training (N2V)
    - Run through a N2V tutorial
    - Denoise your own data with N2V
- Unsupervised training with a noise model (Probabilistic N2V)
    - Run through tutorial
    - Bootstrapping your own noise model
- Bonus: (Hierarchical) Diversity Denoising (DivNoising)
- Bonus: ZeroCostDL4Microscopy

---

## ****Exercise #0: Create a `conda env` and installing all you need in there**

---

Conda environments are useful to keep your workflows running, even if they have conflicting dependencies. For example, one method might be using TensorFlow 1.x and the old python 2.7, while another uses TensorFlow 2.x and some python 3 version.

For the exercises below we will tell you when we suggest creating a new conda environment, but for now, let’s prepare one for CARE, the method we will start playing with in the next exercise.

1. Log in to your VDI, open the terminal and execute the following command 
    
    ```bash
    $ conda create --name care python=3.7
    ```
    
2. This will create a new `conda` environment called `care`, and initialize it with the specified version of python.
3. Activate the newly created environment…
    
    ```bash
    $ conda activate care
    ```
    
4. Now install all you need in this environment. For CARE (the software package is called `csbdeep`), the following should do the trick
    
    ```bash
    $ conda install -c conda-forge cudatoolkit=11.3 cudnn jupyter tensorboard 
    $ conda install jupyter_core=4.6.1
    $ conda install -c conda-forge nb_conda
    $ pip install tensorflow==2.5 csbdeep
    ```
    
5. Clone the CSBDeep repository (into the DL4MIA folder previously created) to get the notebooks:
    
    ```bash
    $ git clone https://github.com/CSBDeep/CSBDeep.git
    ```
    

The packages `tensorboard` and `nb_conda` are not strictly required for using `CSBDeep`, but you will see they are super useful to have them. (We will see the utility of `tensorboard` further below, and `nb_conda` allows you to switch between all installed conda environments in a running jupyter instance… yay!)
So, last but not least, you need to start a jupyter notebook server.

1. In a terminal you open within your VDI, activate the `care` conda environment (see above), then type
    
    ```bash
    $ jupyter notebook
    ```
    
    A browser should automatically open if you run it within your VDI and you are ready to go! If it doesn’t, you can click the link in the terminal.
    
    If it asks for a password, check exercise **00_First_steps** on how to to reset it.
    

**If you want to use your local laptop/computer**, it’s a bit more awkward since we need to do it via `ssh`, which needs some extra commands, but totally worth the effort. If you still want to try, run the following steps:

1. Open a new terminal on your local machine, and `ssh` into your VDI by executing a mad looking command:
    
    ```bash
    $ ssh -L 8888:localhost:8888 <username>@<vdi_ip>
    ```
    
    This connects to your VDI through it’s `<vdi_ip>`, logs in there as `<username>` and forwards your local machines port 8888 to the port 8888 of your VDI. Why? You’ll see soon...
    
2. Connect to your VDI via ssh
    
    ```bash
    $ ssh <username>@<vdi_ip>
    ```
    
3. And activate the `care` conda environment (see above).
4. Now start jupyter.
    
    If you want to use the default port for jupyter (8888), executing the following command should do the trick
    
    ```bash
    $ jupyter notebook --no-browser
    ```
    
    In case you need to use another port, *e.g.* because you want to run two jupyter instances on one machine, you can be more explicit and use…
    
    ```bash
    $ jupyter notebook --no-browser --port=<PORT>
    ```
    
5. Finally, in your browser open the notebook. It should look something like this: *http://localhost:<PORT>/*

If it all worked out, you should now see something like this in your browser:

![Screenshot from 2022-07-13 17-05-36.png](https://file.notion.so/f/s/e714d20c-29e9-4406-a2f8-07d37507798b/Screenshot_from_2022-07-13_17-05-36.png?id=9d56e692-162f-4d9b-9593-3ec3071c1eb0&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=QqsjadQYf9fVZ080pncxbvtlo2k387c9yQeUDdHXcDU&downloadName=Screenshot+from+2022-07-13+17-05-36.png)

### Setting Tensorboard (optional)

Tensorboard is a tool used to visualize the evolution of a neural network throughout the training process. It is a very useful tool to inspect all losses and other parameters in your model. If you want to use it, we will need to start a Tensorboard process:

1. Open a new Terminal window within the VDI and browse to the folder where your current model is being saved. Note that this folder may not exist until you start training a model (below)
    
    ```bash
    $ cd dl4mia/gitrepos/path/to/model/folder/
    ```
    
2. Activate your conda environment once again (as seen above). It should be the same conda environment
    
    ```bash
    $ conda activate care
    ```
    
3. Now we start Tensorboard by executing the following…
    
    ```bash
    $ tensorboard --logdir=. --port 6006
    ```
    
    In truth, you can start Tensorboard from anywhere by putting the whole path to your model in `--logdir`, but it is easier to browse there first and adding `.` which stand for the current directory.
    
    If you notice, we added the `--port` argument as well. It works exactly as seen in the example above for jupyter. Tensorboard needs another port to be forwarded and 6006 is the default one it uses, however, you are free to choose another one as long as it is free.
    

*In case you are not directly using the VDI but are running the notebook via `ssh`:* you may want to use `tensorboard` as well from your local machine. The process is the same as for `jupyter`, open a new terminal in your machine and run the following command:

```bash
$ ssh -L 6006:localhost:6006 <username>@<vdi_ip>
```

---

## Exercise #1: Train your first CARE network (supervised)

---

The GIT repo ‘CSBDeep’ you have previously cloned into ‘~/DL4MIA/CSBDeep’ contains multiple example notebooks:

![Untitled](https://file.notion.so/f/s/e9761db8-a0bc-40a7-9953-edd48748085f/Untitled.png?id=914f06a5-def3-42f6-a772-79ace4f9e166&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=P7041d93yhSaFF6p6LNu7mp-l6Xu9x_XJkSsyO5rmFQ&downloadName=Untitled.png)

Every example will be downloading all required training data and is itself divided into three individual notebooks that need to be executed in sequence. The denoising3D example is a good starting point, but you can really pick any of these if you find another one more interesting and more related to your own data.

> **Note**: Some of the notebooks will have output cells in them. To get the full experience without spoilers, clear all outputs when first opening each notebook. In the jupyter menus, navigate to `Cell > All Outputs > Clear` . You can do the same for other exercises.
> 

**1_datagen.ipynb** - network training usually happens on batches of smaller sized images than the ones recorded on a microscope. Hence, this notebook loads all your image data and chops it into many smaller pieces and stores it into the sub-folder `data` (you can see that folder on the screenshot below, but likely not yet in your own example folder).
Open this notebook, read all explanations, execute all cells, ask any questions that come up - then continue below...

![Untitled](https://file.notion.so/f/s/cdd5c737-2e49-42d6-b303-41a6585b20e0/Untitled.png?id=20a2a85c-b0cf-4304-b6ec-f24bd913546b&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=a7yIvhe3K5XUQDC98hdslkbEiTfa6SOUEOnNt0VRKlI&downloadName=Untitled.png)

**2_training.ipynb** - this notebook will train your CARE network. All outputs will be put into the folder `models`. While you execute this notebook, see what files will be generated inside the models-folder.

As you can see, the example notebooks are containing quite some additional explanations and some cells that have the purpose of showing you the data you are about to use and some sampled results. In this way you can be sure that the right things are happening.
Once you reach the cell that is actually starting the network training, you can go ahead and use Tensorboard (as described in [here](https://www.notion.so/DL4MIA-2023-02-Image-Restoration-exercises-c5cbac04823b4ced8690bbe31c14c832?pvs=21)) to monitor the training process. If you did not change the default settings you will have between 10 and 20 minutes to play with Tensorboard before the training of your first CARE network will be done. Enjoy (and please ask questions if you have any)!
At the very bottom you see, that we can even even export a so-called *Tensorflow Saved Model*. Such a zip file contains all data and metadata to fully define a Tensorflow model, allowing us to use such a saved model to apply a trained CARE network to raw data from within Fiji.

**3_prediction.ipynb** - the last of the three notebooks can be used to apply the network we have previously trained on any dataset you’d like to. Open the notebook and see what it does.
**Note**: if you happen to run into weird out-of-memory errors, you will need to shutdown notebooks that occupy GPU memory but are likely not used any longer. 

One good way to do so is by clicking on the ‘**Running**’-tab in Jupyter, then on the orange ‘**Shutdown**’-button next to the notebooks that are not longer needed:

![Untitled](https://file.notion.so/f/s/3d0bc4cd-4889-49b4-a147-37ef2e763214/Untitled.png?id=ffd74798-ec84-4fcd-b241-d07e2cd1ee2a&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=f2WrMHlky7duAhQK2FYwhxX9rgp41GYfsa3tlRwnRVs&downloadName=Untitled.png)

Alternatively, once you are done running a notebook, you should ‘Close and Halt’ from the *File* menu for the same outcome.

If you still run into memory issues, try using smaller tiles, for example:

```jsx
restored = model.predict(x, axes, n_tiles=(1,32,32))
```

---

## Exercise #2: What is going on behind the scenes?

---

Seek to understand what happened behind the scenes while executing the three notebooks you just executed in Exercise 1.

Try to answer the following questions:

1. Where is the training and test data located?
2. How does data have to be stored so that CSBDeep will find and load it correctly?
3. Where are trained models stored? What models are stored? What is the difference between them?
4. Where in the notebooks is the name of the saved models determined?
5. How can you influence the number of training steps per epoch? What did you use (likely in the interest of time) and what value is suggested?
6. Where can you change the fraction of data being used for `train` and for `validation`? How much did was used in these examples? How much should you use?
7. BONUS: While this is not done in the example notebooks, how would you load an existing CARE network and continue to train it?
8. BONUS: How would you do the same thing in such a way that the additionally trained model is stored separately (under a new name)?

---

## Exercise #3: Train a CARE network “Noise2Noise”

---

> *Data by Reza Shahidi and Gaspar Jekely, Living Systems Institute, Exeter*
> 

> *Notebooks solutions adapted from similar exercises by Larissa Heinrich*
> 

> For this exercise, we have some notebooks available in [https://github.com/dl4mia/02_image_restoration](https://github.com/dl4mia/02_image_restoration) . Feel free to clone this repo.
> 

In this exercise you will start with raw data and decide for yourself how to train a CARE network but using the same scheme as Noise2Noise; using noisy images alone as input and as target. The data contains the same sample, imaged at different levels of noise, so choose carefully which one you should actually use.

**Task**: Use CARE to improve the quality of noisy images without having ground truth.

1. Within the VDI, download the data from [https://tinyurl.com/yxlqqgm2](https://tinyurl.com/yxlqqgm2) and unzip it into the `data` ****directory.
2. The file contains 2 *tiff-stacks*, one for training and one for testing. Open `train.tif` or `test.tif` in Fiji or with python to look at the content. Each stack contains 7 images of the same tissue that were recorded with different scan time settings of a Scanning Electron Microscope (SEM):
    - Image 0 (1 in Fiji) is recorded with 0.2 $\mu$s scan time
    - Image 1 (2 in Fiji) is recorded with 0.5 $\mu$s scan time
    - Image 2 (3 in Fiji) is recorded with 1 $\mu$s scan time
    - Image 3 (4 in Fiji) is recorded with 1 $\mu$s scan time
    - Image 4 (5 in Fiji) is recorded with 2.1 $\mu$s scan time
    - Image 5 (6 in Fiji) is recorded with 5.0 $\mu$s scan time
    - Image 6 (7 in Fiji) is recorded with 5.0 $\mu$s scan time avg. of 4 images
3. Make a copy of the **1_datagen.ipynb**. Rename it to **1_datagenSEM.ipynb**.
    
    **Q**: How would you train a network to denoise images of 1$\mu$s scan time? Which images do you think could be used as input and which as target?
    
4. Open the `training.tif` and save respective images from the stack into the **`train/low`** and **`train/GT`** folders. Use the same name for the input and target images to pair them. You can use Fiji, or the **`imwrite`** function from **`tifffile`:**
    
    ![Untitled](https://file.notion.so/f/s/6ff9be9b-e896-42b3-9e1d-056bff7767ce/Untitled.png?id=8245ef01-fc7a-46fb-aa0e-fe9695e17f07&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=jQddKzD53RShLQdlfHC-Mx-rpbbwnQT__0my6DXyoHA&downloadName=Untitled.png)
    
5. Modify your **1_datagenSEM.ipynb** to work with your data and run it. 
    
    **Q**: You are using 2D images instead of 3D stacks now, what changes?
    
6. Make a copy of **2_training.ipynb**, modify it accordingly and train a network on your data.
7. Make a copy of **3_prediction.ipynb** and modify it accordingly. Open `test.tif`, process it and look at the results for the different acquisition times.
    
    ![Untitled](https://file.notion.so/f/s/0e03a01c-b2da-4794-bd11-569dd8964bd8/Untitled.png?id=4f99dccc-1b97-4aa6-8aa5-ff5050839ee0&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=dGKArwpIAxw7P0_6KZG-AnbS0LB1Q70e0f2kT9qBHuA&downloadName=Untitled.png)
    

- Can you further improve your results by using the data differently or by tweaking the settings?
- How could you train a single network to process all scan times? Be creative! Surprise us!

---

## Exercise #4: Train a Noise2Void network.

---

Noise2Void allows training from single noisy images.

> **Note*:*** you may use your own data if you have some noisy images with you…
> 
1. Clone the Noise2Void repository (for example in the same folder as CSBDeep):
    
    ```bash
    $ git clone [https://github.com/juglab/n2v.git](https://github.com/juglab/n2v.git)
    ```
    
    This repo contains all the sources that make N2V, but you clone it only to get your hands on the tutorials you will find within the exercises folder...
    
2. Activate the care env and install N2V via pip, like so:
    
    ```bash
    $ pip install n2v
    ```
    
    *Side Notes: (i)* installing N2V also into the `care` conda environment is not optimal if you intend to use CARE more often and installation of both packages is not always possible. Some tools you will use might depend on different Tensorflow versions or different and incompatible versions of other dependencies. If you want to use create a conda env for CARE alone, ask Nuno for the version and combination of packages to do so *(ii)* In addition to the installation instructions online, we also installed `tensorboard` and `nb_conda`… Why? Because we love them and want to use them!!! *(iii)* But… wait a sec… we didn’t even install `jupyter`… what is going on? By installing `nb_conda`, `jupyter` is actually a dependency, hence, it got installed for that reason.
    
3. Start `jupyter` and have a look at the `examples/2D/denoising2D_SEM/01_training.ipynb` ****and ****`examples/2D/denoising2D_SEM/02_prediction.ipynb` ****notebooks  in the N2V examples folder. Some things are different from CARE, *i.e.* there is no **01_datagen.ipynb** notebook, but the overall spirit really is the same. Make sure you understand all the steps. Ask us if you don’t.
4. During training (about 15 min): ask us for details you didn’t fully grasp. Or start `tensorboard` ****and follow the learning progress! (If you forgot how, figure it out!)
5. If you have your own data, replace `train.tif` and `test.tif` with your own data. Play with the parameters and have fun. (*Pro tip:* copy any of the exercise folders, name it in a sensible way, then change the contained notebooks to suit your needs…)

**Power Question** (think about it, we’ll discuss it later):

Remember what he heard in the lecture about N2V and how it is trained by taking noisy pixels over to be used as ground-truth. What are the consequences for the loss of the network during training in terms of how we can interpret them? Any clues???

**Bonus Question:** What happens to the loss when your `train` data is significantly smaller than your `validation` data? How does the prediction look?

---

## Exercise #5: Train a Probabilistic N2V network.

---

Probabilistic Noise2Void, just as N2V, allows training from single noisy images.
In order to get some additional quality squeezed out of your noisy input data, PN2V employs an additional noise model which can either be measured directly at your microscope or approximated by a process called ‘bootstrapping’.
Below we will give you a noise model for the first network to train and then bootstrap one, so you can apply PN2V to your own data (in this course we simply don’t have the means to also do the microscopy bits to record a suitable noise model for your data…)

- Note: Our PN2V implementation is written in pytorch, not Keras/TF.
- Note: PN2V experienced multiple “updates” regarding noise model representations. Hence, the original PN2V repository is not any more the one we suggest to use… (despite it of course working just as described in the original publication…)
1. Clone our **PPN2V** repository (the one repo that can do it all): 
    
    ```bash
    $ git clone [https://github.com/juglab/PPN2V.git](https://github.com/juglab/PPN2V.git)
    ```
    
    *Info*: In the readme on GitHub, installing PPN2V is currently suggested via a `yaml` file that specifies all required dependencies. This would mean to do something like the command below (**don’t do it though, go right to the next point**):
    
    ```bash
    $ conda env create -f ppn2vEnvironment.yml
    ```
    
    > **Note:** don’t do it! These instructions do not run in our VDI !!
    > 
2. On our VDI, we run the latest CUDA and the yaml file provided on GitHub leads to trouble (some versions specified in there are too old for CUDA 11). If you try to install and run methods you find on GitHub, these kinds of things happen all the time and are at times very annoying to figure out and fix. In order to get PPN2V to run, we just install stuff by hand
    
    ```bash
    $ conda create -n ppn2v python=3.9
    $ conda activate ppn2v
    $ conda install pytorch torchvision pytorch-cuda=11.8 'numpy<1.24' scipy matplotlib tifffile jupyter -c pytorch -c nvidia
    $ pip install git+https://github.com/juglab/PPN2V.git
    ```
    

1. Note how sneakily I made you install nb_conda again? Why did I not also make you install Tensorboard?

Ok, now that we have installed all we need for (P)PN2V, let’s do something cool with it!
Remember that the difference between N2V and PN2V is that we use a noise model? Importantly, we can record data at the microscope that allows us to get such a noise model. However, we can’t do that during the course, but we can run through an example that comes with all required noise model data:

1. Run through the tutorial in `examples/Convallaria/PN2V`.
2. Note a few things:
    1. There are also folders for CARE and N2V… why the heck would that be? Spoiler: (P)PN2V is the first method we use that is implemented in pytorch instead of Tensorflow and we implemented our own baseline methods also using pytorch so we can compare performances more directly.
    2. Inside the PN2V folder, you find two notebooks for step 1. The first one, 1a, uses recorded images as calibration data to compute a noise model. The second one, 1b, is the one to use if you want to bootstrap a noise model (aka, denoising your data with N2V, then using the noise raw data and the N2V denoised images to compute a noise model). ***We suggest you start with 1a!***
    3. In all the noise model generation notebooks you will find code to create a histogram-based noise model and a GMM based noise model. Strictly speaking, you don’t need to create both, so… make your choice!
    4. You might wanna reduce the number of epochs before you start the training in order to speed things up.
3. Now try PN2V on some of your own data! Oh… and… yes… very likely you don’t have calibration data and need to bootstrap a noise model. For this you can either use N2V from the conda env you installed in the previous exercise, or you use the pytorch N2V version we found just before in the PPN2V repo (this might be simpler since the bootstrapping notebook expects this to have happened)… good luck! Ask us in case of trouble!!!

---

## ****Exercise #6 *(Bonus)*: Train and Use a DivNoising Network**

---

DivNoising is one of the latest unsupervised denoising methods and follows a quite different approach. Instead of a U-Net, DivNoising employs the power of a Variational Auto-Encover (VAE), but adds the use of a noise model to it in a suitable way. 

This approach comes with an extra perk: you will be able to sample diverse interpretations of a noisy image. Why is that cool, you ask? Easy:

- If the diverse samples look all the same (or VERY similar to each other), you know that the noisy data you just denoised is not very ambiguous and you might decide to trust the result more.
- If, on the other hand, the samples look all quite different, you now know that you might not want to trust any of the denoised ‘interpretations’. Additionally we show in the DivNoising paper how diverse samples can be used in meaningful ways for improved downstream analysis of your data.

Since this is a bonus exercise and you are a pro by now, we will keep the instructions here brief, essentially putting you in the position you would find yourself in when you come across a method you find interesting and want to check it out. You can find DivNoising here: [https://github.com/juglab/DivNoising](https://github.com/juglab/DivNoising)

For our VDI, we have to slightly change the installation instructions and have to install all the packages manually. It may look uglier than previous installations, but it is for a good cause!!

1. Run the following commands in order to install DivNoising
    
    ```bash
    $ conda create -n divnoising python=3.7
    $ conda activate divnoising
    $ conda install nb_conda tifffile matplotlib scipy scikit-learn
    $ pip install pytorch-lightning==1.2.10
    ```
    
2. In normal conditions, simply follow the installation instructions you find in the readme on GitHub.
    1. Yes, the author of DivNoising and the readme installs things slightly differently than we are used to by now… just follow his advice, you will be fine…
3. Now start jupyter, just as we did in all the exercises before…
4. Follow the good advice in the readme and run through the Convallaria example

> *Side note:* DivNoising is based on pytorch, but it is using a library called PyTorch Lightning. Why should you care? I tell you: you can use Tensorboard to follow the training progress of your DivNoising network! Yay!
> 

---

## ****Exercise #7 (Bonus): Use ZeroCostDL4Microscopy, e.g. for DecoNoising****

---

This exercise sheet will introduce you to [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki), a toolbox for the training and implementation of common Deep Learning approaches to microscopy imaging. It exploits the ease-of-use and access to GPU provided by [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb).

![Untitled](https://file.notion.so/f/s/37b358f3-d0c4-4dd6-b4ce-0b64e7512df3/Untitled.png?id=5b3bb74b-10fa-45dc-bd31-370161efbc14&table=block&spaceId=10bcea8c-e347-41c2-830b-9cba925c8c74&expirationTimestamp=1697292000000&signature=yL2Hs5UKO-RbM768FKDG7unrBHHy8e07sja4JQVONXs&downloadName=Untitled.png)

---

### Learning Objectives

- Familiarize yourself with Google Colab, and how to use it.
- Familiarize yourself with what ZeroCostDL4Mic (aka ‘Zero’) has to offer.
- Denoise microscopy images using Noise2Void
- [BONUS] Denoise microscopy images using DecoNoising

---

The Google Colaboratory, or "Google Colab" for short, allows you to write and execute Python in your browser, with

- Zero configuration required
- Free access to GPUs
- Easy sharing

Whether you're a student, a data scientist or an AI researcher, Colab can make your work easier. If you’re really lost, you might watch [Introduction to Colab](https://www.youtube.com/watch?v=inN8seMm7UI) to learn more, or just go on reading and working through this exercise sheet.

If you have never ever used Google Colab, please take a few minutes to quickly go over the following notebook (provided from Google for this purpose exactly).

[https://colab.research.google.com/notebooks/intro.ipynb](https://colab.research.google.com/notebooks/intro.ipynb)

So… having access to GPUs from everywhere? Sounds like a dream! This was also what the [people behind Zero](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki#developers-and-testers) thought when they started collecting useful deep learning ideas and casting them into usable notebooks that run on the Google Colab.

A huge advantage is not only that it is free, but all these different models can now be trained using a very similar workflow. Once you go through any of the many Zero notebooks, the others will look similar and take even less time for you to run.

Now… browse through the [long list of notebooks](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki#fully-supported-networks) that are available today. You will spot many ‘old friends’, such as N2V, CARE, etc. Please also note that new notebooks are added quite frequently, so next time you check this page, you might find more useful functionality you might benefit from in your daily work.

**Important steps before you are good to go:**

- You will need a Google account. If you don’t have one, today might be the day to [get one](https://accounts.google.com/signup/v2/webcreateaccount?hl=en&flowName=GlifWebSignIn&flowEntry=SignUp)… (alternatively, you might now focus on the EmbedSeg exercise sheet… ;)

For Google Colab to make the most sense, it is by far the most convenient to store all data on [your Google Drive](https://drive.google.com/drive/my-drive). (It’s not strictly required, but since the alternative is so annoying, we will assume you know how to upload and download data to your Google Drive.

**Run DecoNoising in Zero:**

As mentioned in the lecture, DecoNoising is a fun little idea on top of Noise2Void. (Remember, we introduced a fixed convolution layer just therefore the network output, forcing the deep net to learn some sort of deconvolved image in the layer before that fixed convolution.)

When you look for [DecoNoising on the Zero Wiki](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki#under-beta-testing), you will notice that it is currently under Beta testing. This means two things: *(i)* the notebooks are likely to change a bit over the next weeks or months, and *(ii)* there might be some glitches or smaller issues to encounter when using it. This being said, we used it ourselves many times and it works very well already.

**Now, start the DecoNoising notebook from the [Zero page](https://github.com/HenriquesLab/ZeroCostDL4Mic/wiki#fully-supported-networks), or directly from [here](https://colab.research.google.com/github/HenriquesLab/ZeroCostDL4Mic/blob/master/Colab_notebooks/Beta%20notebooks/DecoNoising_2D_ZeroCostDL4Mic.ipynb).**

***Important Note:*** for the DecoNoising notebook to work, you need at least two training images and all training images need to have the same size in pixels. Hence, if you want, for example, to use the SEM images from one of the previous exercises, you will first have to cut them into smaller images that have the same dimension. If you are lazy, which you really should, feel free to download such crops from [here](https://drive.google.com/file/d/1fAwiw05EKeqnMuBPGzowCu1aYcVJRn8x/view?usp=sharing).

***Bonus Bonus Exercise:*** if you decide to also store the ‘deconvolved’ network output (aka the activation of the network layer just before the fixed convolution layer), try opening that TIFF file in Fiji (it will fail, since it is a 64bit TIFF which Fiji does not understand). You can currently only ‘fix’ this by changing the code in the DecoNoising notebook a bit. If you feel like spending some time… try to fix this! (Spoiler: you need to change the datatype to be e.g. float32…)
