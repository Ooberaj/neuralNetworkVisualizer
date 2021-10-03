<!DOCTYPE html>
<html lang="en-US">
  <head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-F3w7mX95PdgyTmZZMECAngseQB83DfGTowi0iMjiWaeVhAn4FJkqJByhZMI3AhiU" crossorigin="anonymous">
    <link href="styles/style.css" rel="stylesheet">
    <title>Neural Network Visualization</title>
  </head>

  <body>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.1/dist/js/bootstrap.bundle.min.js" integrity="sha384-/bQdsTh/da6pkI1MST/rWKFNjaCP5gBSY4sEBT38Q/9RBh9AH40zEOg7Hlq2THRZ" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/9.4.4/math.min.js" integrity="sha512-OZ6CXzl5JrSc9OM1lxp1OC+zt5gCTVAqy7nWwbdSUE98akAvGl/20WaIqsRUnSpBG+QBkcMkiJVfFvybZ6PtKQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script src="two.min.js"></script>
     <script src="https://kit.fontawesome.com/d171682087.js" crossorigin="anonymous"></script>
    <script src="https://cdn.plot.ly/plotly-2.4.2.min.js"></script>
    <script type="module" src="bundle.js" defer></script>

    <div class="custom-container container px-4">
      <div id="topBar">
        <h1 id="title">Neural Network Visualization</h1>
        <button class="btn btn-info popoverButton" type="button" data-bs-container="body" data-bs-toggle="popover" data-bs-placement="right" data-bs-trigger="focus" data-bs-html='true' data-bs-content="
        <p>Edge Color: A positive weight is represented by a <span style='color: #50C878;'>green edge</span> while a negative weight is represented by a <span style='color: #D2042D;'>red edge</span>.
        <p>Edge Thickness: As positive weights increase and negative weights decrease, edge <b>thickness</b> increases. 
        <p>Node Border Color: <span style='color: #FDDA0D;'>Yellow border</span> represents the node's negative bias while a black border represents positive bias.
        <p>Node Border Thickness: As positive biases increase and negative biases decrease, border <b>thickness</b> increases.
        <p>Node Color: As a node's activation value decreases, its <span style='color: #d3d3d3;'>lightness</span> increases.">Legend</button>
        <button class="btn btn-info popoverButton" type="button" data-bs-container="body" data-bs-toggle="popover" data-bs-placement="right" data-bs-trigger="focus" data-bs-html='true' data-bs-content="
        <p>The neural network is trained on the MNIST dataset to recognize hand-written digits. The network has 784 input neurons with activations loaded from 28x28 images, 30 neurons in 1 hidden layer, and 10 output neurons representing numbers 0 through 9.
        <p>As the visualization is ran, the training process is displayed. The input image for each training example is loaded onto the input layer as the network is trained real-time. Simultaneously, the thickness and color of each edge and node border is changed. However, since the changes are so small, they are difficult to see. The changes are also displayed in each node's tooltip and heatmap.
        ">Description</button>
        <button class="btn btn-info popoverButton" type="button" data-bs-container="body" data-bs-toggle="popover" data-bs-placement="right" data-bs-trigger="focus" data-bs-html='true' data-bs-content="
        <p>Learning Rate = 3.0%
        <p>Mini-batch Size = 30
        <p>Activation function: Sigmoid
        <p>Learning method: Stochastic Gradient Descent
        <p>Training Examples per Epoch: 4000
        <p>Test Examples: 800
        <p>For the output layer, pixel significances are calculated by taking the weighted average of each hidden layer heatmap, where the heatmap's weight is the weight between the hidden layer neuron that corresponds to the heatmap and the output neuron.
        ">Network specifications</button>
        <div id="controls">
          <button type="button" id="slowDown" class="btn btn-outline-secondary custom-btn"><i class="fas fa-fast-backward"></i></button>
          <button type="button" id="startStop" class="btn btn-outline-secondary custom-btn"><i class="fas fa-play"></i><i class="fas fa-pause"></i></button>
          <button type="button" id="speedUp" class="btn btn-outline-secondary custom-btn"><i class="fas fa-fast-forward"></i></button>
          <button type="button" id="prevInputs" class="btn btn-outline-secondary custom-btn" disabled>Previous 60 inputs</button>
          <button type="button" id="nextInputs" class="btn btn-outline-secondary custom-btn">Next 60 inputs</button>
          <p id="inputLayerRange">Viewing input neurons 0 through 60</p>
        </div>
      </div>
      <div class="row">
        <div class="col-8">
          <div id="leftCol" class="col-md-12">
            <div id="network"></div>
          </div>
        </div>
        <div class="col-4 p-4">
          <div id="dataColumn" class="col-md-12">
            <p id="epoch">Epoch: 0</p>
            <p id="trainingExamplesSeen">Training examples seen: 0</p>
            <p id="accuracy">Accuracy = </p>
            <div id="lossFunctionGraph">Click the play button to see cool stuff happen!</div>
            <div id="attributionMap">Hover or click on a node to see more cool stuff happen!</div>
          </div>
        </div>
      </div>
    </div>

  </body>
</html> 
