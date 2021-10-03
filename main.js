var mnist = require('mnist'); 
var gaussian = require('gaussian');
$( document ).ready(function() {
	var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
	var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
	  return new bootstrap.Popover(popoverTriggerEl)
	});
	var popover = new bootstrap.Popover(document.querySelector('#legend'), {
	  trigger: 'focus'
	});
});
let canvasWidth = document.getElementById("leftCol").offsetWidth;
let canvasHeight = (window.innerHeight - document.getElementById("topBar").offsetHeight) * 0.95;
let elem = document.getElementById("network");
let params = { width: canvasWidth, height: canvasHeight };
let two = new Two(params).appendTo(elem);
let frame = two.update().frameCount;

function Network(numNodes){
	// Constructor of neural network class to initialize the neural network.
	this.numLayers = numNodes.length;
	this.numNodes = numNodes;
	// Construct a bias matrix. Each row represents a non-input layer and containing that layer's biases.
	// Bias values are randomly initialized
	this.biases = [];
	for(let i = 1; i < this.numLayers; i++){
		let distribution = gaussian(0, 1);
		this.biases.push([]);
		for(let p = 0; p < this.numNodes[i]; p++){
			this.biases[i-1].push(distribution.ppf(Math.random()));
		}
	}
	// Construct a weight array that contains a weight matrix for each layer. Each row of a weight matrix
	// represents a distinct node. Each entry is the weight of an incoming edge for that node.
	this.weights = [];
	for(let i = 1; i < this.numLayers; i++){
		let distribution = gaussian(0, 1/Math.pow(numNodes[i], 0.25));
		this.weights.push([]);
		for(let p = 0; p < this.numNodes[i]; p++){
			this.weights[i-1].push([]);
			for(let q = 0; q < this.numNodes[i-1]; q++){
				this.weights[i-1][p].push(distribution.ppf(Math.random()));
		    }
		}
	}
	this.nodes = [];
	this.edges = [];
	this.activations = [];
	this.nodeGroups = [];
	this.trainingExamplesSeen = 0;
	this.examplesSeenOverTime = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
}

Network.prototype.feedForward = function(activations){
	//Requires vector of activations for the input layer and returns the vector of activations for the output layer.
	for (let i = 0; i < this.numLayers - 1; i++){
		activations = sigmoid(math.add(math.multiply(this.weights[i], activations), this.biases[i]));
	}
	return activations;
}

Network.prototype.updateMiniBatch = function(miniBatch, learningRate, costGradientBiases, costGradientWeights){
	//Update the neural network's weights and bias using gradient descent
	for(let i = 0; i < miniBatch.length; i++){
		this.trainingExamplesSeen++;
		let exampleGradient = this.backprop(miniBatch[i].input, miniBatch[i].output, copyArray(costGradientBiases), copyArray(costGradientWeights));
		costGradientBiases = addArrays(costGradientBiases, exampleGradient[0]);
		costGradientWeights = addArrays(costGradientWeights, exampleGradient[1]);
	}
	//turn costGradientBiases into an array of row vectors instead of column vectors
	for(let layer = 0; layer < this.numLayers - 1; layer++){
		costGradientBiases[layer] = math.transpose(costGradientBiases[layer])
	}
	this.biases = addArrays(this.biases, multiplyArrays(costGradientBiases, -(learningRate/miniBatch.length)));
	this.weights = addArrays(this.weights, multiplyArrays(costGradientWeights, -(learningRate/miniBatch.length)));

	for(let i = 1; i < 11; i++){
		this.examplesSeenOverTime.push(i + this.trainingExamplesSeen);
	}
	lossFunctionGraph(loss, this.examplesSeenOverTime);
	if(activeNode[0] != null){
		//update attributionMap
		Plotly.purge("attributionMap");
		attributionMap(activeNode[0], activeNode[1]);

		//update tooltip
		if(!inactive){
			let layerNum = activeNode[0];
			let nodeNum = activeNode[1];
			let nodeId = activeNode[2];
			let bias = net.biases[layerNum - 1][nodeNum];
			let activation = net.activations[layerNum][nodeNum];
			let toolTipString = "Bias:\n" + bias + "\nActivation:\n" + activation;
			$(nodeId).tooltip('dispose');
			$(nodeId).attr("data-bs-toggle", "tooltip");
			$(nodeId).attr("title", toolTipString).tooltip('show');
			$(nodeId).attr("data-trigger", "manual");
			$(nodeId).tooltip('show');
		}
	}
	let examplesSeenText = "Training examples seen: " + this.trainingExamplesSeen;
	$("#trainingExamplesSeen").text(examplesSeenText);
}

Network.prototype.evaluate = function(testData){
	let sum = 0;
	for(let i = 0; i < testData.length; i++){		
		//some of the test data holds input data that's undefined. The try catch block skips it.
		try { 
			let netOutput = this.feedForward(testData[i].input);
			let realOutput = testData[i].output;
			if(i<5){console.log(netOutput + " / " + netOutput.indexOf(math.max(netOutput))); console.log(realOutput + " / " + realOutput.indexOf(math.max(realOutput)));}
			if(netOutput.indexOf(math.max(netOutput)) == realOutput.indexOf(math.max(realOutput))){sum++;}
		}
		catch (e) {
			console.error(e.message);
		}
	}
	return sum;
}

Network.prototype.costDerivative = function(outputActivations, actualOutput){
	/* Returns a vector containing the partial derivatives of the cost in terms of activation values for the
	   output layer. Cost is defined as the difference between calculated and desired activations.
	*/
	return math.subtract(outputActivations, actualOutput);
}

Network.prototype.backprop = function(input, actualOutput, costGradientBiases, costGradientWeights){
	//feedforward
	let activations = [input];
	let zs = []; 
	for (let i = 0; i < this.numLayers - 1; i++){
		// z is the value that is inputted into the activation function. no value for input layer.
		let z = math.add(math.multiply(this.weights[i], activations[i]), this.biases[i]);
		zs.push(z);
		activations.push(sigmoid(z));
	}
	this.activations = activations;
	//backward pass
	let error = math.transpose([math.dotMultiply(this.costDerivative(activations[activations.length - 1], math.transpose(actualOutput)), sigmoidDerivative(zs[zs.length - 1]))]);
	costGradientBiases[costGradientBiases.length - 1] = error;
	//turn 1x1 error vector into a scalar to avoid dimension problems in matrix multiplication
	error = toScalar(error);
	costGradientWeights[costGradientWeights.length - 1] = math.multiply(error, [activations[activations.length - 2]]);
	for(let i = this.numLayers - 2; i > 0; i--){
		let z = zs[i-1];
		//index i instead of i + 1 because the weight matrix's length is numlayers - 1 due to the exclusion of the input layer
		error = math.dotMultiply(math.multiply(math.transpose(this.weights[i]), error), math.transpose([sigmoidDerivative(z)]));
		costGradientBiases[i - 1] = error; 
		costGradientWeights[i - 1] = math.multiply(error, [activations[i - 1]]); //[] turns vector into matrix for library's dimension weirdness
	}
	loss.push(MSE(activations[activations.length - 1], actualOutput));
	return [costGradientBiases, costGradientWeights];
}

Network.prototype.SGD = function(trainingData, learningRate, epochs, miniBatchSize, testData, costGradientBiases, costGradientWeights, epoch, i){
	//Neural network is trained using stochastic gradient descent with mini-batches. Training data is an array 
	//of tuples: (training input, desired result).
	//initialize the cost gradient for biases and weights that will be eventually be passed into the backprop algorithm.
	
	//document.getElementById("network").onload = function continueSGD() { 
	this.updateMiniBatch(trainingData.slice(i, i + miniBatchSize), learningRate, costGradientBiases, costGradientWeights);
}

Network.prototype.duplicateNetworkStructure = function(){
	let costGradientBiases = [];
	let costGradientWeights = [];
	for(let i = 1; i < this.numLayers; i++){
		costGradientBiases.push([]);
		for(let p = 0; p < this.numNodes[i]; p++){
			costGradientBiases[i-1].push([0]);
		}
	}
	for(let i = 1; i < this.numLayers; i++){
		costGradientWeights.push([]);
		for(let p = 0; p < this.numNodes[i]; p++){
			costGradientWeights[i-1].push([]);
			for(let q = 0; q < this.numNodes[i-1]; q++){
				costGradientWeights[i-1][p].push(0);
		    }
		}
	}
	return [costGradientBiases, costGradientWeights];
}

//Misc Math functions
function MSE(calculatedOutputs, actualOutput){
	//Mean-sqared error
	return Math.pow(math.norm(math.subtract(calculatedOutputs, actualOutput)), 2)/net.trainingExamplesSeen;	
}

function sigmoidDerivative(z){
	// Reutrns the derivative of the sigmoid function.
	return math.dotMultiply(sigmoid(z), math.subtract(1, sigmoid(z))); 
}

function sigmoid(matrix){
	//apply sigmoid activation function element wise to the matrix
	return math.map(matrix, x => 1.0/(1.0+Math.exp(-x)));
}

//Misc Matrix functions
function toScalar(vector){
	//turns a 1x1 vector into a scalar
	if(vector.length == 1){
		return vector[0];
	}else{
		return vector;
	}
}

function addArrays(a, b){
	//matrix addition but if different rows had different lengths 1-dimension deep
	let result = [];
	for(let i = 0; i < a.length; i++){
		//[i][0] to de-matrixfy. Vector is contained in an array and needs to be taken out for addition to work.
		if(b[i].length == 1 && Array.isArray(b[i][0])){
			result.push(math.add(a[i], b[i][0]));
		}else{
			result.push(math.add(a[i], b[i]));
		}	
	}
	return result;
}

function multiplyArrays(a, b){
	//matrix multiplication but if different rows had different lengths 1-dimension deep
	let result = []; 
	if(typeof b === 'number'){
		for(let i = 0; i < a.length; i++){
			result.push(math.multiply(a[i], b));
		}
	}
	return result;
}

function copyArray(arr){
	//creates a duplicate of at an atmost 3-dimensional array
	let copy = [];
	if(Array.isArray(arr[0][0])){
	//3-dim
		for(let i = 0; i < arr.length; i++){
			copy.push([]);
			for(let p = 0; p < arr[i].length; p++){
				copy[i].push([...arr[i][p]]);
			}
		}
	}else{
	//2-dim
		for(let i = 0; i < arr.length; i++){
			copy.push([...arr[i]]);
		}
	}
	return copy;
}

function loadData(trainingExamples, tests){
	let set = mnist.set(trainingExamples, tests);
	let trainingSet = set.training;
	let testSet = set.test;
	return [trainingSet, testSet];
}

function drawNetwork(){
	//draw each layer
	//draw weights between each layer
	let network = [];
	let inputRadius = 4;
	let hiddenRadius = 7;
	let outputRadius = 8;
	let interLayerDistance = canvasWidth/(net.numNodes.length - 1)

	//draw input layer
	network.push(drawLayer(inputsDisplayed, inputRadius, data[0][0].input, 6, canvasHeight - (inputRadius * 2), 0));

	//draw hidden layers
	for(let layer = 1; layer < net.numNodes.length - 1; layer++){
		network.push(drawLayer(net.numNodes[layer], hiddenRadius, data[0][0].input, interLayerDistance * layer, canvasHeight - (hiddenRadius * 2), layer));
	}

	//draw output layer
	let outputIndex = net.numNodes.length - 1;
	network.push(drawLayer(net.numNodes[outputIndex], outputRadius, data[0][0].input, interLayerDistance * outputIndex - (outputRadius * 2), canvasHeight - (outputRadius * 2), outputIndex));
	net.nodes = network;
	drawEdges();
}

function drawLayer(numNodes, radius, activations, xPos, canvasHeight, layerNum){
	// radius, interdistance, xPos, canvasHeight: px
	//fillColor is an array of activations nodes should be filled with a grayscale color associated with its activation value. 
	//Very dark represents high activation value while very light represents low activation value. 
	let layer = [];
	let nodeGroup = two.makeGroup();
	for(let nodeNum = 0; nodeNum < numNodes; nodeNum++){
		let node = two.makeCircle(xPos, canvasHeight - nodeNum * (canvasHeight/numNodes), radius);
		//allows getting the node's layer and node number by accessing its id property.
		node.fill = 'white';
		if(layerNum != 0){
			let bias = net.biases[layerNum - 1][nodeNum];
			if(bias > 0){
				node.stroke = '#FDDA0D';
				node.linewidth = 2 * bias;
			}else if(bias == 0){
				node.noStroke();
			}else{
				node.linewidth = 2 * Math.abs(bias);
			}
		}else{
			node.linewidth = 1;
		}
		layer.push(node);
		nodeGroup.add(node);
	}
	net.nodeGroups.push(nodeGroup);
	two.update();
	return layer;
}

function drawEdges(){
	//draw weights between current node and all nodes in previous layers
	let edges = [];
	let edgeGroup = two.makeGroup();
	for(let currLayer = 1; currLayer < net.numNodes.length; currLayer++){
		edges.push([]);
		let prevLayer = currLayer - 1;
		for(let currNode = 0; currNode < net.nodes[currLayer].length; currNode++){
			edges[currLayer - 1].push([]);
			for(let prevLayerNode = 0; prevLayerNode < net.nodes[prevLayer].length; prevLayerNode++){
				let edge = two.makeLine(net.nodes[prevLayer][prevLayerNode].position.x, 
											net.nodes[prevLayer][prevLayerNode].position.y, 
											net.nodes[currLayer][currNode].position.x, net.nodes[currLayer][currNode].position.y);
				edge.opacity = 0.1;
				let weight = net.weights[currLayer - 1][currNode][prevLayerNode];
				if(weight < 0){
					edge.stroke = '#D2042D';
				}else if(weight == 0){
					edge.stroke = 'black';
				}else{
					edge.stroke = '#50C878';
				}
				edge.linewidth = Math.abs(weight);
				edges[currLayer - 1][currNode].push(edge);
				edgeGroup.add(edge);
			}
		}
	}
	net.edges = edges;
	two.update();
}

function updateNodes(){
	let newNodes = [];
	for(let layer = 0; layer < net.nodes.length; layer++){
		newNodes.push([]);
		for(let nodeNum = 0; nodeNum < net.nodes[layer].length; nodeNum++){
			let node = net.nodes[layer][nodeNum];
			if(layer == 0){
				node.fill = activationToColor(net.activations[layer][nodeNum + inputNodeCount]);
			}else{
				node.fill = activationToColor(net.activations[layer][nodeNum]);
				let bias = net.biases[layer - 1][nodeNum];
				if(bias > 0){
					node.stroke = '#FDDA0D';
					node.linewidth = 2 * bias;
				}else if(bias == 0){
					node.noStroke();
				}else{
					node.linewidth = 2 * Math.abs(bias);
				}
			}
			newNodes[layer].push(node);
		}
	}
	net.nodes = newNodes;
	two.update();
}

function updateEdges(){
	//draw weights between current node and all nodes in previous layers
	let newEdges = [];
	for(let currLayer = 1; currLayer < net.numNodes.length; currLayer++){
		newEdges.push([]);
		let prevLayer = currLayer - 1;
		for(let currNode = 0; currNode < net.nodes[currLayer].length; currNode++){
			newEdges[currLayer - 1].push([]);
			for(let prevLayerNode = 0; prevLayerNode < net.nodes[prevLayer].length; prevLayerNode++){
				let edge = net.edges[currLayer - 1][currNode][prevLayerNode];
				let weight = net.weights[currLayer - 1][currNode][prevLayerNode];
				if(prevLayer == 0){
					weight = net.weights[currLayer - 1][currNode][prevLayerNode + inputNodeCount];
				}
				if(weight < 0){
					edge.stroke = '#D2042D';
				}else if(weight == 0){
					edge.stroke = 'black';
				}else{
					edge.stroke = '#50C878';
				}
				edge.linewidth = Math.abs(weight);
				newEdges[currLayer - 1][currNode].push(edge);
			}
		}
	}
	net.edges = newEdges;
	two.update();
}

function updateInputs(inputLayerChange = 0, preTraining = false){
	let inputLayer = [];
	for(let nodeNum = 0; nodeNum < net.nodes[0].length; nodeNum++){
		let node = net.nodes[0][nodeNum];
		if(!preTraining){
			node.fill = activationToColor(net.activations[0][nodeNum + inputLayerChange]);
		}
		inputLayer.push(node);
	}
	net.nodes[0] = inputLayer;
	two.update();
}

function updateHiddenEdges(inputLayerChange = 0, preTraining = false){
	let newEdges = [];
	for(let currNode = 0; currNode < net.nodes[1].length; currNode++){
		newEdges.push([]);
		for(let prevLayerNode = 0; prevLayerNode < net.nodes[0].length; prevLayerNode++){
			let edge = net.edges[0][currNode][prevLayerNode];
			let weight = net.weights[0][currNode][prevLayerNode + inputLayerChange];
			if(weight < 0){
				edge.stroke = '#D2042D';
			}else if(weight == 0){
				edge.stroke = 'black';
			}else{
				edge.stroke = '#50C878';
			}
			edge.linewidth = Math.abs(weight);
			newEdges[currNode].push(edge);
		}
	}
	net.edges[0] = newEdges;
	two.update();
}

function activationToColor(activation){
	// Conver a positve activation value to a grayscale color code. The higher the activation, the darker i.e. less light 
	//the shade of black.
	return 'hsl(240, 0%, ' + (100 - (activation * 100)) + '%)';
}

function makeActive(node, layerNum, inputLayerLength){
	//make incoming edges opaque
	let nodeId = node.id;
	let delimiter = nodeId.indexOf('-');
	//use two.js's automatically created id to get the node's index in the edge matrix.
	let nodeNum = parseInt(nodeId.substring(delimiter + 1, nodeId.length)) - inputLayerLength;
	let incomingEdges = net.edges[layerNum - 1][nodeNum];
	for(let i = 0; i < incomingEdges.length; i++){
		incomingEdges[i].opacity = 1;
	}
	two.update();

	//make tooltip
	nodeId = "#" + node.id;
	$(nodeId).attr("data-bs-toggle", "tooltip");
	let bias = net.biases[layerNum - 1][nodeNum];
	let activation = net.activations[layerNum][nodeNum];
	let toolTipString = "Bias:\n" + bias + "\nActivation:\n" + activation;
	$(nodeId).attr("title", toolTipString);
	$(nodeId).attr("data-trigger", "manual");
	$(nodeId).tooltip('show');
	inactive = false;

	attributionMap(layerNum, nodeNum);

	activeNode = [layerNum, nodeNum, nodeId];
}

function makeInactive(node, layer, inputLayerLength){
	//reset opacity of incoming edges
	let nodeId = node.id;
	let delimiter = nodeId.indexOf('-');
	//use two.js's automatically created id for the node to get its node index in the edge matrix.
	let nodeNum = parseInt(nodeId.substring(delimiter + 1, nodeId.length)) - inputLayerLength;
	let incomingEdges = net.edges[layer - 1][nodeNum];
	for(let i = 0; i < incomingEdges.length; i++){
		incomingEdges[i].opacity = 0.1;
	}
	nodeId = "#" + node.id;
	$(nodeId).tooltip('hide');
	inactive = true;
	two.update();
}

function interactiveNodes(layerId, layerNum, numPrevChildren){
	$(layerId).children().each(function () {
		let nodeId = '#' + this.id;
		let active = false;
		let clicked = false;
		$(nodeId).mouseenter(function(){
			document.getElementById('attributionMap').innerHTML = "";
			makeActive(this, layerNum, numPrevChildren);
			active = true;
		});
		$(nodeId).click(function(){
			if(clicked == true){
				makeInactive(this, layerNum, numPrevChildren);
				active = false;
				clicked = false;
			}else{
				makeActive(this, layerNum, numPrevChildren);;
				active = true;
				clicked = true;
			}
		});
		$(nodeId).mouseleave(function(){
			if(!clicked){
				makeInactive(this, layerNum, numPrevChildren);
				active = false;
			}
		});
	});
}

function setLastEpoch(val){
	lastEpoch = val;
}

function setDataIndex(val){
	dataIndex = val;
}

function lossFunctionGraph(loss, examplesSeenOverTime){
	let lossFunctionGraph = document.getElementById('lossFunctionGraph');
	var layout = {
		title: {
		  	text: 'Mean-squared Error Over Training',
		  	font: {
		  		family: $("body").css("font-family"),
		  		size: 14,
		  		color: '#444444'
		  	},
		  	xref: 'canvas',
		  	x: 0.55,
  		},
		xaxis: {
			title:{
				text: 'Number of Examples Trained on',
			  	font: {
				 	family: $("body").css("font-family"),
				  	size: 12,
				  	color: '#444444'
		  		}
			}
	  	},
		yaxis: {
			title:{
				text: 'Mean-squared Error',
			  	font: {
				 	family: $("body").css("font-family"),
				  	size: 12,
				  	color: '#444444'
		  		}
			}
	 	},
	  	width: document.getElementById("dataColumn").offsetWidth,
	  	height: canvasHeight * 0.3,
	  	margin: {l: 50, r: 0, t: 20, b: 30}
	};
	let fig = Plotly.newPlot(lossFunctionGraph, [{x: examplesSeenOverTime, y: loss}], layout, {scrollZoom: true});
}

function attributionMap(layerNum, nodeNum){
	let z = 0;
	let titleText = '';
	if(layerNum == 1){
		z = hiddenLayerAttributions(layerNum - 1, nodeNum);
		titleText = 'Weight between each Input Neuron(Pixel)<br>and Hidden Layer Neuron ' + nodeNum;
	}else{
		z = outputLayerAttributions(nodeNum);
		titleText = 'Significance of Input Neurons(Pixels)<br>on Output Neuron ' + nodeNum;
	}
	let width = document.getElementById("dataColumn").offsetWidth;
	let layout = {
	  title: {
	  	text: titleText,
	  	font: {
	  		family: $("body").css("font-family"),
	  		size: 14,
	  		color: '#444444'
	  	},
	  	yref: "container",
	  	y: 0.9,
	  },
	  xaxis: {
	      showgrid: false,
	      zeroline: false,
	      visible: false
	  },
	  yaxis: {
	      showgrid: false,
	      zeroline: false,
	      visible: false
	  },
	  margin: {l: width * 0.1, r: 0, t: 60, b: 40},
	  width: width * 0.8,
	  height: width * 0.8
	};
	let data = [
	  {
	    z: z,
	    type: 'heatmap'
	  }
	];
	Plotly.newPlot('attributionMap', data, layout, {displayModeBar: false});
	$("#attributionMap").css({"margin-left": width * 0.1});
}

function hiddenLayerAttributions(layerNum, nodeNum){
	let weights = net.weights[layerNum][nodeNum];
	let dividedWeights = [];
	for(let i = 0; i < weights.length; i+= 28){
		dividedWeights.push([]);
		for(let q = i; q < i + 28; q++){
			dividedWeights[i/28].push(weights[q]);
		}
	}
	console.log(dividedWeights);
	return dividedWeights;
}

function outputLayerAttributions(nodeNum){
	let pixelSignificances = [];
	let count = -1;
	for(let i = 0; i < net.numNodes[0]; i++){
		if(i%28 == 0){
			pixelSignificances.push([]);
			count++;
		}
		let pixelWeights = [];
		for(let q = 0; q < net.numNodes[1]; q++){
			pixelWeights.push(net.weights[0][q][i]);
		}
		let pixelSignificance = math.sum(math.dotMultiply(pixelWeights, net.weights[1][nodeNum]))/30;
		pixelSignificances[count].push(pixelSignificance);
	}
	return pixelSignificances;
}

let learningRate = 3.0;
let miniBatchSize = 30;
let epochs = 100;
let lastEpoch = 0;
let dataIndex = 0;
let inputNodeCount = 0;
let activeNode = [null, null]
let loss = [];
let inputsDisplayed = 56;
let inactive = true;
let speed = 350;

let net = new Network([784, 30, 10]);
let a = net.duplicateNetworkStructure();
let costGradientBiases = a[0];
let costGradientWeights = a[1];
let data = loadData(4000, 800);
let test = data[1][0];
let trainingData = data[0];
drawNetwork();

let hiddenLayerId = '#' + net.nodeGroups[1].id;
let outputLayerId = '#' + net.nodeGroups[2].id;
interactiveNodes(hiddenLayerId, 1, 59);
interactiveNodes(outputLayerId, 2, 90);

var playing = false;
$("#startStop").click(function(){
	if(playing == true){
		playing = false;
	}else{
		playing = true;
		SGD(lastEpoch, dataIndex);
		document.getElementById('lossFunctionGraph').innerHTML = "";
	}
});

$("#prevInputs").click(function(){
	if(lastEpoch == 0 && dataIndex == 0){
		updateInputs(-inputsDisplayed + inputNodeCount, true);
	}else{
		updateInputs(-inputsDisplayed + inputNodeCount);
	}
	updateHiddenEdges(-inputsDisplayed + inputNodeCount);
	inputNodeCount -= inputsDisplayed;

	let inputRangeText = "Viewing input neurons " + inputNodeCount + " through " + (inputNodeCount + inputsDisplayed) + ".";
	$('#inputLayerRange').text(inputRangeText);

	$("#nextInputs").prop("disabled", false);

	if(inputNodeCount == 0){
		$("#prevInputs").prop("disabled", true);
	}
});

$("#nextInputs").click(function(){
	if(lastEpoch == 0 && dataIndex == 0){
		updateInputs(inputsDisplayed + inputNodeCount, true);
	}else{
		updateInputs(inputsDisplayed + inputNodeCount);
	}
	updateHiddenEdges(inputsDisplayed + inputNodeCount);
	inputNodeCount += inputsDisplayed;

	let inputRangeText = "Viewing input neurons " + inputNodeCount + " through " + (inputNodeCount + inputsDisplayed) + ".";
	$('#inputLayerRange').text(inputRangeText);

	$("#prevInputs").prop("disabled", false);

	if(inputNodeCount == 728){
		$("#nextInputs").prop("disabled", true);
	}
});

$("#speedUp").click(function(){
	if(speed > 0){
		speed -= 50;
	}
});

$("#slowDown").click(function(){
	speed += 100;
});
