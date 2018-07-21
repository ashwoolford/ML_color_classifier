
let data;

let model;
let xs, ys;

let lossP;
let labelP;

let rSlider, bSlider, gSlider;

let labelList = [
	'red-ish',
	'green-ish',
	'blue-ish',
	'orange-ish',
	'yellow-ish',
	'pink-ish',
	'purple-ish',
	'brown-ish',
	'grey-ish'

]

function preload(){

  data = loadJSON('colorData.json'); 

}

function setup() {
	//console.log(data.entries.length);

	labelP = createP('');

	lossP = createP('Loss');
	

	rSlider = createSlider(0,255,255);
	gSlider = createSlider(0,255,255);
	bSlider = createSlider(0,255,0);

	let colors = [];
	let labels = [];
	for(let record of data.entries){
		let col = [record.r / 255, record.g / 255, record.b / 255];
		colors.push(col);
		labels.push(labelList.indexOf(record.label));
	}

	//console.log(colors);
	xs = tf.tensor2d(colors);
	

	let labelsTensor = tf.tensor1d(labels, 'int32');
	//labelsTensor.print();

	ys = tf.oneHot(labelsTensor, 9);

	labelsTensor.dispose();



	//xs.print();
	//ys.print();


	model = tf.sequential();

	let hidden  = tf.layers.dense({
		units: 9,
		activation: 'sigmoid',
		inputDim: 3
	});


	let output = tf.layers.dense({
		units: 9,
		activation: 'softmax'
	});

	model.add(hidden);
	model.add(output);

	//create an optimizer

	const lr = 0.2;
	const optimizer = tf.train.sgd(lr);

	model.compile({

		optimizer: optimizer,
		loss: 'categoricalCrossentropy'
	});


	train().then(results=> {
		console.log(results.history.loss);
	});
}

async function train(){

	const options = {
		epochs: 20,
		validationSplit: 0.1,
		shuffle: true,
		callbacks: {
			onTrainBegin: () => console.log('training started'), 
			onTrainEnd: () => console.log('training completed'),
			onBatchEnd: tf.nextFrame,
			onEpochEnd: (num, logs) => {
				console.log('Epoch: ' + num);
				lossP.html('Loss: ' + logs.loss);

			}
		}
	}

	return await model.fit(xs, ys, options);
}

function draw(){

	let r = rSlider.value();
	let g = gSlider.value();
	let b = bSlider.value();
	background(r, g, b);

	const xs = tf.tensor2d([
		[r/255, g/255, b/255]
	]);

	let results = model.predict(xs);
	let index = results.argMax(1).dataSync()[0];

	let label = labelList[index];
	labelP.html(label);

	//index.print();






}