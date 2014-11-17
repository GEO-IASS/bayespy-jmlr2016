﻿using System;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using System.Diagnostics;

namespace MicrosoftResearch.Infer.Tutorials
{

/*
    public class BayesianPCAFullyFactorizedModel
    {
        // Inference engine
        public InferenceEngine engine = null;
        // Model variables
        public Variable<int> vN = null;
        public Variable<int> vD = null;
        public Variable<int> vM = null;
        public VariableArray2D<double> vData = null;
        //* DEFAULT SCALAR VERSION
        public VariableArray2D<double> vW = null;
        public VariableArray2D<double> vZ = null;
        public VariableArray2D<double> vT = null;
        public VariableArray2D<double> vU = null;
        public VariableArray<double> vMu = null;
        public VariableArray<double> vPi = null;
        public VariableArray<double> vAlpha = null;
        // Priors - these are declared as distribution variables
        // so that we can set them at run-time. They are variables
        // from the perspective of the 'Random' factor which takes
        // a distribution as an argument.
        public Variable<Gamma> priorAlpha = null;
        public Variable<Gaussian> priorMu = null;
        public Variable<Gamma> priorPi = null;
        // Model ranges
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

        /// <summary>
        /// Model constructor
        /// </summary>
        public BayesianPCAFullyFactorizedModel()
            {

                // Stuff for the lower bound / evidence
                IfBlock block = Variable.If(evidence);

                // Start model
                    
                // The various dimensions will be set externally...
                N = Variable.New<int>().Named("NumSamples");
                M = Variable.New<int>().Named("NumDimensions");
                D = Variable.New<int>().Named("NumComponents");
                rN = new Range(vN).Named("N");
                rD = new Range(vD).Named("D");
                rM = new Range(vM).Named("M");
                
                // ... as will the data
                //vData = Variable.Array<double>(rN, rD).Named("data");
                // ... and the priors
                //priorAlpha = Variable.New<Gamma>().Named("PriorAlpha");
                priorMu = Variable.New<Gaussian>().Named("PriorMu");
                priorPi = Variable.New<Gamma>().Named("PriorPi");
                
                // Mixing matrix
                W = Variable.Array<Vector>(M).Named("W");
                Alpha = Variable.WishartFromShapeAndRate(2,
                                                         PositiveDefiniteMatrix.IdentityScaledBy(D,
                                                                                                 2.0));
                W[rM] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                   Alpha).ForEach(rM);

                // Latent variables
                Z = Variable.Array<Vector>(N).Named("Z");
                Z[rN] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                   PositiveDefiniteMatrix.Identity(D)).ForEach(rN);

                // Multiply
                ZtimesW = Variable.Array<double>(N, D).Named("ZtimesW");
                ZtimesW[rM, rN] = Variable.InnerProduct(W[rM], Z[rN]);

                // Bias
                mu = Variable.Array<double>(M).Named("mu");
                mu[rM] = Variable.Random<double, Gaussian>(priorMu).ForEach(rM);
                ZWplusMu = Variable.Array<double>(rM, rN).Named("ZWplusMu");
                ZWplusMu[rM, rN] = ZtimesW[rM, rN] + mu[rM];

                // Observation noise
                //pi = Variable.Array<double>(r
                pi = Variable.Random<double, Gamma>(priorPi);

                // Observations
                data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZWplusMu[rM,rN], pi);


                // Mixing matrix. Each row is drawn from a Gaussian with zero mean and
                // a precision which will be learnt. This is a form of Automatic
                // Relevance Determination (ARD). The larger the precisions become, the
                // less important that row in the mixing matrix is in explaining the data
                vAlpha = Variable.Array<double>(rM).Named("Alpha");
                vW = Variable.Array<double>(rM, rD).Named("W");
                vAlpha[rM] = Variable.Random<double, Gamma>(priorAlpha).ForEach(rM);
                vW[rM, rD] = Variable.GaussianFromMeanAndPrecision(0, vAlpha[rM]).ForEach(rD);
                // Latent variables are drawn from a standard Gaussian
                vZ = Variable.Array<double>(rN, rM).Named("Z");
                vZ[rN, rM] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(rN, rM);
                // Multiply the latent variables with the mixing matrix...
                vT = Variable.MatrixMultiply(vZ, vW).Named("T");
                // ... add in a bias ...
                vMu = Variable.Array<double>(rD).Named("mu");
                vMu[rD] = Variable.Random<double, Gaussian>(priorMu).ForEach(rD);
                vU = Variable.Array<double>(rN, rD).Named("U");
                vU[rN, rD] = vT[rN, rD] + vMu[rD];
                // ... and add in some observation noise ...
                vPi = Variable.Array<double>(rD).Named("pi");
                vPi[rD] = Variable.Random<double, Gamma>(priorPi).ForEach(rD);
                // ... to give the likelihood of observing the data
                vData[rN, rD] = Variable.GaussianFromMeanAndPrecision(vU[rN, rD], vPi[rD]);

                // End evidence/lowerbound block
                block.CloseBlock();
                        
                // Inference engine
                engine = new InferenceEngine(new VariationalMessagePassing());

                        
                return;
            }
    }
//*/

    public class BayesianPCAModel
    {
        // Inference engine
        public InferenceEngine engine = null;
        // Model variables
        public VariableArray<Vector> W = null;
        public VariableArray<Vector> Z = null;
        public VariableArray<double> mu = null;
        public VariableArray2D<double> data = null;
        public VariableArray<double> pi = null;
        //*/
        //public VariableArray2D<double> vT = null;
        //public VariableArray2D<double> vU = null;
        //public VariableArray<double> vMu = null;
        //public VariableArray<double> vPi = null;
        //public VariableArray<double> vAlpha = null;
        // Priors - these are declared as distribution variables
        // so that we can set them at run-time. They are variables
        // from the perspective of the 'Random' factor which takes
        // a distribution as an argument.
        //public Variable<Gamma> priorAlpha = null;
        //public Variable<Gaussian> priorMu = null;
        public Variable<Gamma> priorPi = null;
        // Model ranges
        //public Variable<int> N = null;
        //public Variable<int> D = null;
        //public Variable<int> M = null;
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

        /// <summary>
        /// Model constructor
        /// </summary>
        public BayesianPCAModel(int M, int N, int D)
            {

                // Stuff for the lower bound / evidence
                IfBlock block = Variable.If(evidence);

                // Start model
                    
                // The various dimensions will be set externally...
                rN = new Range(N).Named("N");
                rD = new Range(D).Named("D");
                rM = new Range(M).Named("M");
                
                // ... as will the data
                //vData = Variable.Array<double>(rN, rD).Named("data");
                // ... and the priors
                //priorAlpha = Variable.New<Gamma>().Named("PriorAlpha");
                //priorMu = Variable.New<Gaussian>().Named("PriorMu");
                priorPi = Variable.New<Gamma>().Named("PriorPi");
                //priorPi.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);
                
                // Mixing matrix
                W = Variable.Array<Vector>(rM).Named("W");
                var Alpha = Variable.WishartFromShapeAndRate(rD.SizeAsInt,
                                                             PositiveDefiniteMatrix.IdentityScaledBy(rD.SizeAsInt,
                                                                                                     rD.SizeAsInt));
                W[rM] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(rD.SizeAsInt),
                                                                   Alpha).ForEach(rM);

                // Latent variables
                Z = Variable.Array<Vector>(rN).Named("Z");
                Z[rN] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(rD.SizeAsInt),
                                                                   PositiveDefiniteMatrix.Identity(rD.SizeAsInt)).ForEach(rN);

                // Multiply
                var ZtimesW = Variable.Array<double>(rM, rN).Named("ZtimesW");
                ZtimesW[rM, rN] = Variable.InnerProduct(W[rM], Z[rN]);

                // Bias
                /*
                mu = Variable.Array<double>(rM).Named("mu");
                mu[rM] = Variable.Random<double, Gaussian>(priorMu).ForEach(rM);
                var ZWplusMu = Variable.Array<double>(rM, rN).Named("ZWplusMu");
                ZWplusMu[rM, rN] = ZtimesW[rM, rN] + mu[rM];
                //*/

                // Observation noise
                pi = Variable.Array<double>(rM).Named("pi");
                pi[rM] = Variable.Random<double, Gamma>(priorPi).ForEach(rM);

                // Observations
                data = Variable.Array<double>(rM, rN).Named("Y");
                data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZtimesW[rM,rN], pi[rM]);
                //data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZWplusMu[rM,rN], pi[rM]);

                // End evidence/lowerbound block
                block.CloseBlock();
                        
                // Inference engine
                engine = new InferenceEngine(new VariationalMessagePassing());

                return;
            }

    }

    /// <summary>
    /// Run a Bayesian PCA example
    /// </summary>
    [Example("Applications", "A Bayesian Principal Components Analysis example")]
    public class BayesianPCA
    {
        static int numComp = 10;
        static int numFeat = 100;
        static int numObs = 1000;

        static void Main()
            {
                    
                Console.Write("Running PCA.\n");
                BayesianPCA bpca = new BayesianPCA();
                bpca.Run();
                return;
            }

        public double cputime()
            {
                Process process = Process.GetCurrentProcess();
                //System.TimeSpan lifeInterval = (DateTime.Now - process.StartTime);
                return process.TotalProcessorTime.TotalMilliseconds;
                //FILETIME ftSysIdle, ftSysKernel, ftSysUser;
                //FILETIME ftProcCreation, ftProcExit, ftProcKernel, ftProcUser;
                //GetProcessTimes(GetCurrentProcess(), &ftProcCreation, &ftProcExit, &ftProcKernel, &ftProcUser);
                
            }
            
        public void Run()
            {
                int M = numFeat;
                int N = numObs;
                int D = numComp;
                
                BayesianPCAModel bpca = new BayesianPCAModel(M, N, D);
                if (!(bpca.engine.Algorithm is VariationalMessagePassing))
                {
                    Console.WriteLine("This example only runs with Variational Message Passing");
                    return;
                }
	
                // Set a stable random number seed for repeatbale runs
                //Rand.Restart(12347);
                double[,] data = generateData();
                // Set the data
                bpca.data.ObservedValue = data;
                // Set the dimensions
                //bpca.N.ObservedValue = numObs;
                //bpca.M.ObservedValue = numFeat;
                //bpca.D.ObservedValue = numComp;
                // Set the priors
                //bpca.priorMu.ObservedValue = Gaussian.FromMeanAndPrecision(0.0, 0.01);
                bpca.priorPi.ObservedValue = Gamma.FromShapeAndRate(0.01, 0.01);
                //bpca.priorAlpha.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);
                // Initialize the W marginal to break symmetry
                bpca.W.InitialiseTo(RandomGaussianVectorArray(M, D));
                double logEvidence = 0;

                double t0, t1;
                t0 = cputime();

                for (int j=0; j < 100; j++)
                {
                    bpca.engine.NumberOfIterations = j+1;
                    // Infer the marginals
                    //bpca.engine.Infer<IDistribution<double[,]>>(bpca.W);
                    //bpca.engine.Infer<IDistribution<double[,]>>(bpca.Z);
                    //bpca.engine.Infer<IDistribution<double[]>>(bpca.mu);
                    //bpca.engine.Infer<IDistribution<double[]>>(bpca.pi);
                    // Show lower bound
                    logEvidence = bpca.engine.Infer<Bernoulli>(bpca.evidence).LogOdds;
                    Console.WriteLine("The lower bound: {0}", logEvidence);
                    t1 = cputime();
                    Console.WriteLine("Time {0}\n", t1-t0);
                    t0 = t1;
                }
                /*
                IDistribution<double[,]> wMarginal = bpca.engine.Infer<IDistribution<double[,]>>(bpca.W);
                //IDistribution<double[]> muMarginal = bpca.engine.Infer<IDistribution<double[]>>(bpca.mu);
                IDistribution<double[]> piMarginal = bpca.engine.Infer<IDistribution<double[]>>(bpca.pi);
                // Convert distributions over arrays of doubles to arrays of distributions
                //Gaussian[,] inferredW = Distribution.ToArray<Gaussian[,]>(wMarginal);
                //Gaussian[] inferredMu = Distribution.ToArray<Gaussian[]>(muMarginal);
                Gamma[] inferredPi = Distribution.ToArray<Gamma[]>(piMarginal);
                // Print out the results
                Console.WriteLine("Inferred W:");
                printMatrixToConsole(inferredW);
                Console.Write("Mean absolute means of rows in W: ");
                printVectorToConsole(meanAbsoluteRowMeans(inferredW));
                Console.Write("    True bias: ");
                printVectorToConsole(trueMu);
                Console.Write("Inferred bias: ");
                printVectorToConsole(inferredMu);
                Console.Write("    True noise:");
                printVectorToConsole(truePi);
                Console.Write("Inferred noise:");
                printVectorToConsole(inferredPi);
                Console.Write("Distribution of W\n" + bpca.engine.Infer(bpca.W));
                Console.WriteLine();
                //*/

            }

        /// <summary>
        /// True W. Inference will find a different basis
        /// </summary>
        static double[,] trueW = new double[numComp, numFeat];
        //{
        // {-0.30, 0.40, 0.20, -0.15, 0.20, -0.25, -0.50, -0.10, -0.25, 0.10},
        //{-0.10, -0.20, 0.40, 0.50, 0.15, -0.35, 0.05, 0.20, 0.20, -0.15},
        //{ 0.15, 0.05, 0.15, -0.10, -0.15, 0.25, -0.10, 0.15, -0.30, -0.55},
        //};

        /// <summary>
        /// True bias
        /// </summary>
        static double[] trueMu = new double[numFeat];
        //static double[] trueMu = { -0.95, 0.75, -0.20, 0.20, 0.30, -0.35, 0.65, 0.20, 0.25, 0.40 };

        ///// <summary>
        ///// True observation noise
        ///// </summary>
        //static double[] truePi = { 1.00, 2.00, 4.00, 2.00, 1.00, 2.00, 3.00, 4.00, 2.00, 1.00 };
        /// <summary>
        /// True observation noise
        /// </summary>
        static double[] truePi = new double[numFeat];
        //static double[] truePi = {  8.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 11.0 };

        public static IDistribution<Vector[]> RandomGaussianVectorArray(int N, int C)
            {
                VectorGaussian[] array = new VectorGaussian[N];
                for (int i = 0; i < N; i++)
                {
                    Vector mean = Vector.Zero(C);
                    for (int j = 0; j < C; j++) mean[j] = Rand.Normal();
                    array[i] = new VectorGaussian(mean, PositiveDefiniteMatrix.Identity(C));
                }
                return Distribution<Vector>.Array(array);
            }
        
        /// <summary>
        /// Generate data from the true model
        /// </summary>
        /// <param name="numObs">Number of observations</param>
        static double[,] generateData()
            {
                // Generate true parameters
                for (int i=0; i < numComp; i++)
                {
                    for (int j=0; j < numFeat; j++)
                    {
                        trueW[i,j] = Gaussian.Sample(0.0, 1.0);
                    }
                }
                for (int j=0; j < numFeat; j++)
                {
                    trueMu[j] = 0.0;
                    truePi[j] = 1.0;
                }
                //int numComp = trueW.GetLength(0);
                //int numFeat = trueW.GetLength(1);
                // Generate data matrix
                double[,] data = new double[numFeat, numObs];
                Matrix WMat = new Matrix(trueW);
                Vector z = Vector.Zero(numComp);
                for (int i=0; i < numObs; i++)
                {
                    // Sample scores from standard Gaussian
                    for (int j=0; j < numComp; j++)
                        z[j] = Gaussian.Sample(0.0, 1.0);
                    // Mix the components with the true mixture matrix
                    Vector t = z * WMat;
                    for (int j=0; j < numFeat; j++)
                    {
                        // Add in the bias
                        double u = t[j] + trueMu[j];
                        // ... and the noise
                        data[j, i] = Gaussian.Sample(u, truePi[j]);
                    }
                }
                return data;
            }

        /// <summary>
        /// Create an array of Gaussian distributions with random mean and unit variance
        /// </summary>
        /// <param name="row">Number of rows</param>
        /// <param name="col">Number of columns</param>
        /// <returns>The array as a distribution over a 2-D double array domain</returns>
        private static IDistribution<double[,]> randomGaussianArray(int row, int col)
            {
                Gaussian[,] array = new Gaussian[row, col];
                for (int i = 0; i < row; i++)
                {
                    for (int j = 0; j < col; j++)
                    {
                        array[i, j] = Gaussian.FromMeanAndVariance(Rand.Normal(), 1);
                    }
                }
                return Distribution<double>.Array(array);
            }

        /// <summary>
        /// Mean absolute row means
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        private static double[] meanAbsoluteRowMeans(Gaussian[,] matrix)
            {
                double[] mam = new double[matrix.GetLength(0)];
                double mult = 1.0 / ((double)matrix.GetLength(1));
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    double sum = 0.0;
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        sum += Math.Abs(matrix[i, j].GetMean());
                    mam[i] = mult * sum;
                }
                return mam;
            }

        /// <summary>
        /// Print the means of a 2-D array of Gaussians to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printMatrixToConsole(Gaussian[,] matrix)
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        Console.Write("{0,5:0.00}\t", matrix[i, j].GetMean());
                    Console.WriteLine("");
                }
            }

        /// <summary>
        /// Print a 2-D double array to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printMatrixToConsole(double[,] matrix)
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        Console.Write("{0,5:0.00}\t", matrix[i, j]);
                    Console.WriteLine("");
                }
            }

        /// <summary>
        /// Print the means of a 1-D array of Gaussians to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printVectorToConsole(Gaussian[] vector)
            {
                for (int i = 0; i < vector.GetLength(0); i++)
                    Console.Write("{0,5:0.00}\t", vector[i].GetMean());
                Console.WriteLine("");
            }

        /// <summary>
        /// Print the means of a 1-D array of Gammas to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printVectorToConsole(Gamma[] vector)
            {
                for (int i = 0; i < vector.GetLength(0); i++)
                    Console.Write("{0,5:0.00}\t", vector[i].GetMean());
                Console.WriteLine("");
            }

        /// <summary>
        /// Print a 1-D double array to the console
        /// </summary>
        /// <param name="matrix"></param>
        private static void printVectorToConsole(double[] vector)
            {
                for (int i = 0; i < vector.GetLength(0); i++)
                    Console.Write("{0,5:0.00}\t", vector[i]);
                Console.WriteLine("");
            }
    }
}
