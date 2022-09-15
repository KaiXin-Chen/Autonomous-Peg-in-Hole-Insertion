# Autonomous Peg in Hole insertion
## Example Result
(double clicking this image will lead you to a Youtube video)
[![Watch the video](https://github.com/KaiXin-Chen/Autonomous-Peg-in-Hole-Insertion/blob/master/setup.PNG)](https://www.youtube.com/watch?v=mbyZ9o9rxJU)
## Overview
<br>This is a class project for Stanford CS229 (machine learning) and CS231N (deep learning for computer vision) class. The contributers are Hao Li, Kaixin Chen, Tianheng Shi (ranked in alphabetical order). Special thanks to Stanford vision and learning lab for providing their demonstration dataset and borrowing us the robot arm for testing.
<br>For detained problem statements, emplementation and training details and results please read the report submitted for the two classes. However, a very brief summery of the project is provided below.
<br>We want the robot arm to perform imitation learning (behavior cloning) with images captured by a third person view camera as the only input. The data set we used is demonstrations (action at each time step and cooresponding images captued by the same third person view camera) of this robot arm performing peg insertion (these demonstrations are obtained by a person manipulating the robot). We first built and trained a basesline. This baseline is a ResNet 18 (vision encoder module) concatenated with a FC net (controller module), and in the report for CS229, we presented our efforts to make improvments to the controller module and in the report for CS231N, we presented our effort to make improvements to the vision encoder.
