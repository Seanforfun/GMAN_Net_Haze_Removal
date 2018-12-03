# Haze removal(Jan, 2018 - June, 2018)
### Introduction
We are using a new convolutional neural network "GMEAN" to create a End-to-end haze removal network. We feed the network hazed images and we can get the clear result directly. According to our results, we get the PSNR for 28.189dB and MSE for 0.964 on [RESIDES](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2) out door dataset.And our results on indoor evaluation dataset for PSNR and mse are 20.527dB and 0.8081 repectively.
![Imgur](https://i.imgur.com/HfPpj6Q.png)

### Demonstration
Only listed several examples, more results can be found in my [github](https://github.com/Seanforfun/Deep-Learning/tree/master/DehazeNet/Results).
#### Outdoor
<table>
	<tr>
		<th>Hazy</th>
		<th>Groundtruth</th>
		<th>Our result</th>	
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/8S6cpRe.jpg"/></th>		
		<th><img src="https://i.imgur.com/fUhQuld.png"/></th>
		<th><img src="https://i.imgur.com/jOLhygU.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/deNcv4O.jpg"/></th>		
		<th><img src="https://i.imgur.com/L66qRbw.png"/></th>
		<th><img src="https://i.imgur.com/miDdMgk.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/rFTGcVD.jpg"/></th>		
		<th><img src="https://i.imgur.com/aSmMOJE.png"/></th>
		<th><img src="https://i.imgur.com/r1YPHym.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/iBE5sGw.jpg"/></th>		
		<th><img src="https://i.imgur.com/u6HY6qE.png"/></th>
		<th><img src="https://i.imgur.com/x2Uu3Tc.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/cVtaJnm.jpg"/></th>		
		<th><img src="https://i.imgur.com/4QKZdHa.png"/></th>
		<th><img src="https://i.imgur.com/wQ6SmiQ.jpg"/></th>
	</tr>
</table>

#### Indoor
<table>
	<tr>
		<th>Hazy</th>
		<th>Groundtruth</th>
		<th>Our result</th>	
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/81MUWBh.png"/></th>		
		<th><img src="https://i.imgur.com/bsqSWNC.png"/></th>
		<th><img src="https://i.imgur.com/pBhsVG8.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/UrDTN2G.png"/></th>		
		<th><img src="https://i.imgur.com/75yuyRw.png"/></th>
		<th><img src="https://i.imgur.com/Y7TbUOR.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/rx5jrpd.png"/></th>		
		<th><img src="https://i.imgur.com/7cPB8Wg.png"/></th>
		<th><img src="https://i.imgur.com/fFIwaMG.jpg"/></th>
	</tr>
	<tr>
		<th><img src="https://i.imgur.com/9bWE6zj.png"/></th>		
		<th><img src="https://i.imgur.com/fbAWMTg.png"/></th>
		<th><img src="https://i.imgur.com/r6GiyXj.jpg"/></th>
	</tr>	
</table>

1. Reference: [Generic Model-Agnostic Convolutional Neural Network for Single Image Dehazing](https://arxiv.org/abs/1810.02862)
2. Code work: [GMEAN Code](https://github.com/Seanforfun/Deep-Learning/tree/master/DehazeNet)
3. Co-worker: [Zheng Liu](https://github.com/MintcakeDotCom)