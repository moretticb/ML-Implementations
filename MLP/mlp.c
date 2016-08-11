/*
 * +-----------------------------------------------------------+
 * |     MULTILAYER PERCEPTRON ARTIFICIAL NEURAL NETWORK       |
 * |      Backpropagation algorithm  with momentum term        |
 * |  Implemented by CAIO BENATTI MORETTI - www.moretticb.com  |
 * |                     COMPILABLE IN GCC                     |
 * +-----------------------------------------------------------+
 * +---------------------------------+
 * |  Program USAGE - TRAINING MODE  |
 * +---------------------------------+
 * 
 * $ cat inputFile | ./mlp -i INPUTS -o OUTPUTS -l LAYERS n1 n2 n3 [-[e | E | W]]
 *         WHERE:
 *         "mlp" is the executable version of this code
 *         "-i" indicates the number of INPUTS (3 inputs: -i 3)
 *         "-o" indicates the number of OUTPUTS (1 output: -o 1)
 *         "-l" indicates the number of LAYERS and the size of each layer
 *                 i.e. 2 LAYERS (first - 5 neurons; second - 1 neuron): -l 2 5 1
 *         -[e | E | W]
 *                 "e" verboses the number of epochs
 *                 "E" verboses the EQMs of each epoch
 *                 "W" does not verbose adjusted weights
 * 
 * EXAMPLE OF inputFile:
 * 
 * ---------inputFile--------
 * 3
 * 2.0000 1.0000 0.3240 1 0 0
 * 0.0040 0.2380 2.0003 0 1 0
 * 7.0050 0.7500 2.0001 0 0 1
 * -----END OF inputFile-----
 * 
 * Line 1   : the number of instances
 * Line 2-n : instance following this format: x1 x2 x3 d1 d2 d3
 *         WHERE:
 *         x1...x3 are the inputs
 *         d1...d3 is the desired output of the respective neuron in the output layer
 * 
 * 
 * +----------------------------------+
 * |  Program USAGE - OPERATION MODE  |
 * +----------------------------------+
 * $ ./mlp -i INPUTS -o OUTPUTS -l LAYERS n1 n2 nLAYERS -w
 *         WHERE:
 *         "-w" indicates the insertion of the adjusted weights
 *              for classification
 * 
 *  Runtime verbose may guide the user for inserting weights and
 * perform classification/regression
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>

#define RATE 0.1
#define EPSLON 0.0000005
#define MOMENTUM 0.8

typedef struct {
	int weights;
	double *w, *lw;
} NEURON;

typedef struct {
	int size;
	NEURON *n;
} LAYER;

typedef struct {
	int inputs, layers;
	LAYER *l;
} NETWORK;

NETWORK *createNetwork(int, int, int*);
double *getI(NETWORK *, int, double *);
double *getY(NETWORK *, int, double *);
int isLastLayer(NETWORK *, int);
double quadraticError(NETWORK *, double *, double *);
double avgQuadError(NETWORK *, double *, int, double *);
double *getDelta(NETWORK *, double *, double *, int);
double sech(double);
double activation(double x);
double activationDeriv(double x);
double drand();
void adjustWeights(NETWORK *, int, double *, double *, double *, double *);
void checkArgs(int, char **, int *, int *, int *, int *, int **, int *, double **, int *, int *, int *);
void showWeights(NETWORK *);
void showUsage();

time_t rseed=0;
int wCount = 0, initW = 0;
double *wValues;

int main(int argc, char **argv){
	int i, j, c=0, inputs, outputs, samples=-1, epoch=0, layers, *npl, showEpochs=0, notShowWeights=0, showEqms=0;
	double diff;

	checkArgs(argc, argv, &inputs, &outputs, &samples, &layers, &npl, &initW, &wValues, &showEqms, &showEpochs, &notShowWeights);

	if(argc <= 1)
		showUsage();

	if(initW > 0)
		printf("Insert the %d weights:\n", initW);
	for(i=0;i<initW;i++)
		scanf("%lf", &wValues[i]);

	NETWORK *n = createNetwork(inputs, layers, npl);

	if(initW == 0){
		if(samples == -1)
			scanf("%d", &samples);

		double **sample = (double **) malloc(sizeof(double*)*samples);
		double **desired = (double **) malloc(sizeof(double*)*samples);

		while(c<samples){
			*(sample+c) = (double *) malloc(sizeof(double)*inputs);
			*(desired+c) = (double *) malloc(sizeof(double)*outputs);

			sample[c][0] = -1.0;
			for(i=1;i<=inputs;i++)
				scanf("%lf", &sample[c][i]);

			for(i=0;i<outputs;i++)
				scanf("%lf", &desired[c][i]);

			c++;
		}

		if(showEqms)
			printf("EQM:\n");
	
		do {
			diff = avgQuadError(n, sample[0], samples, desired[0]);

			for(i=0;i<samples;i++){
				for(j=n->layers-1; j>=0; j--){
					double *delta = getDelta(n, sample[i], desired[i], j);
					double *prevY = j>0 ? getY(n, j-1, sample[i]) : NULL;
					adjustWeights(n, j, sample[i], desired[0], delta, prevY);
				}
			}

			diff = fabs(diff - avgQuadError(n, sample[0], samples, desired[0]));
			epoch++;

			if(showEqms)
				printf("%.16lf\n", diff);

		} while(diff > EPSLON);

		if(showEqms)
			printf("--\n\n");

		if(!notShowWeights){
			printf("Weights:\n");
			showWeights(n);
			printf("--\n\n");
		}

		if(showEpochs)
			printf("Epochs: %d\n", epoch);

	} else {

		double *input = (double *) malloc(sizeof(double)*(inputs+1));

		while(1){
			printf("Insert the input: ");
			input[0] = -1.0;
			for(i=1;i<=inputs;i++)
				scanf("%lf", &input[i]);
			
			double *out = getY(n, layers-1, input);
			
			for(i=0;i<outputs;i++)
				printf("y%d: %lf\t", i, out[i]);

			printf("\n---\n");
		}

	}

	return 0;
}


NETWORK *createNetwork(int inputs, int layers, int *layerNeurons){
	NETWORK *n = (NETWORK *) malloc(sizeof(NETWORK));
	n->layers = layers;
	n->inputs = inputs;
	n->l = (LAYER *) malloc(sizeof(LAYER)*layers);
	
	int i;
	for(i=0;i<layers;i++){

		int neurons = layerNeurons[i];

		LAYER *currLayer = n->l+i;

		currLayer->size = neurons;
		currLayer->n = (NEURON *) malloc(sizeof(NEURON)*neurons);

		int j;
		for(j=0;j<neurons;j++){
			NEURON *neuron = currLayer->n+j;
			neuron->weights = (i==0?n->inputs:layerNeurons[i-1])+1;
			neuron->w = (double *) malloc(sizeof(double)*neuron->weights);
			neuron->lw = (double *) malloc(sizeof(double)*neuron->weights);

			int k;
			for(k=0;k<neuron->weights;k++){
				neuron->w[k] = drand();
				neuron->lw[k] = 0.0;
			}
		}

	}

	return n;
}

void showWeights(NETWORK *n){
	int i, j, k, layers = n->layers, total=0;
	for(i=0;i<layers;i++){
		LAYER *l = n->l+i;
		for(j=0;j<l->size;j++){
			NEURON *neuron = l->n+j;
			for(k=0;k<neuron->weights;k++){
				printf("%lf\n", *(neuron->w+k));
				total++;
			}
		}
	}
}

double sech(double x){
	return 1.0/cosh(x);
}

double activation(double x){
	//Hyperbolic Tangent
	//return tanh(x);
	
	//Logistic
	return 1.0/(1.0+exp(-x));
}

double activationDeriv(double x){
	//Derivative of Tanh
	//return sech(x)*sech(x);
	//return 1.0-pow(activation(x),2);
	
	//Derivative of Logistic
	return activation(x)*(1.0-activation(x));
}

double drand(){
	if(wCount < initW)
		return wValues[wCount++];

        if(rseed==0){
                time(&rseed);
                srand(rseed);
        }
        return (double)rand()/(double)RAND_MAX;
}

int isLastLayer(NETWORK *n, int layer){
	return layer>=n->layers-1;
}

double *getI(NETWORK *n, int layer, double *sample){
	LAYER *currLayer = n->l+layer;
	int i, j, ISize = currLayer->size, wLen = currLayer->n->weights;
	double *I = (double *) malloc(sizeof(double)*ISize);

	double *prevI = layer<1 ? sample : getY(n, layer-1, sample);

	for(i=0;i<ISize;i++){
		I[i] = 0.0;
		NEURON *currNeuron = currLayer->n+i;
		for(j=0;j<wLen;j++){
			I[i] += currNeuron->w[j] * prevI[j];
		}
	}

	return I;
}

double *getY(NETWORK *n, int layer, double *sample){
	LAYER *currLayer = n->l+layer;
	int i, last = isLastLayer(n, layer), YSize = currLayer->size+!last;
	double *I = getI(n, layer, sample);
	double *Y = (double *) malloc(sizeof(double)*YSize);

	for(i=0;i<YSize;i++)
		if(last)
			Y[i] = activation(I[i]);
		else
			Y[i] = i==0?-1.0:activation(I[i-1]);

	return Y;
}

double quadraticError(NETWORK *n, double *sample, double *desired){
	double e = 0.0, *Y = getY(n, n->layers-1, sample);
	int i, outputs = (n->l+n->layers-1)->size;

	for(i=0;i<outputs;i++)
		e += pow(desired[i] - Y[i], 2);

	return e*0.5;
}

double avgQuadError(NETWORK *n, double *samples, int sLen, double *desired){
	int i, j;
	double eqm = 0.0;

	for(i=0;i<sLen;i++)
		eqm += quadraticError(n, samples+i*n->inputs, desired+i*(n->l+n->layers-1)->size);
	
	return eqm/(double) sLen;
}

double *getDelta(NETWORK *n, double *sample, double *desired, int layer){
	int i, layerSize = (n->l+layer)->size; 
	double *delta = (double *) malloc(sizeof(double) * layerSize);
	double *I = getI(n, layer, sample);

	if(isLastLayer(n, layer)){
		double *Y = getY(n, layer, sample);

		for(i=0;i<layerSize;i++)
			delta[i] = (desired[i] - Y[i]) * activationDeriv(I[i]);
		
	} else {
		int j;
		double *nextDelta = getDelta(n, sample, desired, layer+1);
		LAYER *nextLayer = n->l+layer+1;

		for(i=0;i<layerSize;i++){
			delta[i]=0.0;

			for(j=0; j < nextLayer->size; j++)
				delta[i] += nextDelta[j] * *((nextLayer->n+j)->w+i+1);

			delta[i] *= activationDeriv(I[i]);
		}

	}

	return delta;
}

void adjustWeights(NETWORK *n, int layer, double *sample, double *desired, double *delta, double *prevY){
	LAYER *l = n->l+layer;
	int i, j, layerSize = l->size;

	for(i=0;i<layerSize;i++){
		NEURON *neuron = l->n+i;

		for(j=0;j<neuron->weights;j++){
			NEURON *neuron = l->n+i;
		}

		for(j=0;j<neuron->weights;j++){
			//momentum term
			double term = MOMENTUM * (*(neuron->w+j) - *(neuron->lw+j));
			*(neuron->lw+j) = *(neuron->w+j);
			//conventional backpropagation with momentum term
			*(neuron->w+j) = *(neuron->w+j) + term + RATE * delta[i] * (layer==0?sample[j]:prevY[j]);
		}

	}
}

void checkArgs(int argc, char **argv, int *inputs, int *outputs, int *samples, int *layers, int **npl, int *initW, double **wValues, int *showEqms, int *showEpochs, int *notShowWeights){
	int i;
	for(i=1;i<argc;i++){
		char c=argv[i][0], p=argv[i][1];
		if(c == '-')
			if(p == 'o'){
				*outputs = atoi(argv[++i]);
			} else if(p == 'i'){
				*inputs = atoi(argv[++i]);
			} else if(p == 's'){
				*samples = atoi(argv[++i]);
			} else if(p == 'l'){
				int j, *n;
				*layers = atoi(argv[++i]);
				n = (int *) malloc(sizeof(int)*(*layers));
				for(j=0;j<*layers;j++)
					n[j] = atoi(argv[++i]);
				*npl = n;
			} else if(p == 'w'){
				int j, total=(*inputs)+1, *n = *npl;
				double *w;
				for(j=0;j<*layers;j++){
					if(j==0)
						total *= n[0];
					else
						total += (n[j-1]+1)*n[j];
				}
				*initW = total;
				w = (double *) malloc(sizeof(double)*total);
				*wValues = w;
			} else if(p == 'E' || p == 'e' || p == 'W'){
				int j;
				for(j=1;j<strlen(argv[i]);j++){
					char param = argv[i][j];
					if(param == 'E')
						*showEqms = 1;
					else if(param == 'e')
						*showEpochs = 1;
					else if(param == 'W')
						*notShowWeights = 1;
				}
			}
	}
}

void showUsage(){
	printf("+-----------------------------------------------------------+\n");
	printf("|     MULTILAYER PERCEPTRON ARTIFICIAL NEURAL NETWORK       |\n");
	printf("|      Backpropagation algorithm  with momentum term        |\n");
	printf("|  Implemented by CAIO BENATTI MORETTI - www.moretticb.com  |\n");
	printf("|                     COMPILABLE IN GCC                     |\n");
	printf("+-----------------------------------------------------------+\n");
	
	printf("+---------------------------------+\n");
	printf("|  Program USAGE - TRAINING MODE  |\n");
	printf("+---------------------------------+\n\n");

	printf("$ cat inputFile | ./mlp -i INPUTS -o OUTPUTS -l LAYERS n1 n2 n3 [-[e | E | W]]\n");
	printf("\tWHERE:\n");
	printf("\t\"mlp\" is the executable version of this code\n");
	printf("\t\"-i\" indicates the number of INPUTS (3 inputs: -i 3)\n");
	printf("\t\"-o\" indicates the number of OUTPUTS (1 output: -o 1)\n");
	printf("\t\"-l\" indicates the number of LAYERS and the size of each layer\n");
	printf("\t\ti.e. 2 LAYERS (first - 5 neurons; second - 1 neuron): -l 2 5 1\n");
	printf("\t-[e | E | W]\n");
	printf("\t\t\"e\" verboses the number of epochs\n");
	printf("\t\t\"E\" verboses the EQMs of each epoch\n");
	printf("\t\t\"W\" does not verbose adjusted weights\n\n");

	printf("EXAMPLE OF inputFile:\n\n");

	printf("---------inputFile--------\n");
	printf("3\n");
	printf("2.0000 1.0000 0.3240 1 0 0\n");
	printf("0.0040 0.2380 2.0003 0 1 0\n");
	printf("7.0050 0.7500 2.0001 0 0 1\n");
	printf("-----END OF inputFile-----\n\n");

	printf("Line 1   : the number of instances\n");
	printf("Line 2-n : instance following this format: x1 x2 x3 d1 d2 d3\n");
	printf("\tWHERE:\n");
	printf("\tx1...x3 are the inputs\n");
	printf("\td1...d3 is the desired output of the respective neuron in the output layer\n\n\n");


	printf("+----------------------------------+\n");
	printf("|  Program USAGE - OPERATION MODE  |\n");
	printf("+----------------------------------+\n");

	printf("$ ./mlp -i INPUTS -o OUTPUTS -l LAYERS n1 n2 nLAYERS -w\n");
	printf("\tWHERE:\n");
	printf("\t\"-w\" indicates the insertion of the adjusted weights\n\t     for classification\n");
	printf("\t\n");
printf(" Runtime verbose may guide the user for inserting weights and\nperform classification/regression\n");
	exit(0);
}
