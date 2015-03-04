using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

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
        public Variable<PositiveDefiniteMatrix> Alpha = null;
        public VariableArray<Vector> W = null;
        public VariableArray<Vector> Z = null;
        public VariableArray<double> mu = null;
        public VariableArray2D<double> data = null;
        public Variable<double> tau = null;
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

        public BayesianPCAModel(int M, int N, int D)
            {

                // Stuff for the lower bound / evidence
                IfBlock block = Variable.If(evidence);

                // The various dimensions will be set externally...
                rN = new Range(N).Named("N");
                rD = new Range(D).Named("D");
                rM = new Range(M).Named("M");
                
                // NOTE: Compared to BayesPy, prior parameters are multiplied by
                // 0.5 because of different parameterization of the Wishart
                // distribution.
                Alpha = Variable.WishartFromShapeAndRate(0.5*D,
                                                         PositiveDefiniteMatrix.IdentityScaledBy(D, 0.5*D));
                // Mixing matrix
                W = Variable.Array<Vector>(rM).Named("W");
                //W[rM] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                //                                                    PositiveDefiniteMatrix.Identity(D)).ForEach(rM);
                W[rM] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                    Alpha).ForEach(rM);

                // Latent variables
                Z = Variable.Array<Vector>(rN).Named("Z");
                Z[rN] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                    PositiveDefiniteMatrix.Identity(D)).ForEach(rN);

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
                // Set the priors
                //bpca.priorPi.ObservedValue = Gamma.FromShapeAndRate(0.01, 0.01);
                //bpca.priorAlpha.ObservedValue = Gamma.FromShapeAndRate(2.0, 2.0);
                //priorPi = Variable.New<Gamma>().Named("PriorPi");
                //pi = Variable.Array<double>(rM).Named("pi");
                tau = Variable.GammaFromShapeAndRate(0.01, 0.01);
                //pi[rM] = Variable.Random<double, Gamma>(priorPi).ForEach(rM);

                // Observations
                data = Variable.Array<double>(rM, rN).Named("Y");
                data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZtimesW[rM,rN], tau);

                // End evidence/lowerbound block
                block.CloseBlock();
                        
                // Inference engine
                engine = new InferenceEngine(new VariationalMessagePassing());

                return;
            }

    }

    public class BayesianPCA
    {

        static void Main(string[] args)
            {
                    
                Console.Write("Running PCA.\n");
                BayesianPCA bpca = new BayesianPCA();
                bpca.Run(int.Parse(args[0]),  // components
                         int.Parse(args[1]),  // seed
                         int.Parse(args[2])); // maxiter
                return;
            }

        public double get_cputime()
            {
                Process process = Process.GetCurrentProcess();
                return process.TotalProcessorTime.TotalMilliseconds;
            }
            
        public void Run(int D, int seed, int maxiter)
            {
                
                // Restart the infer.NET random number generator
                Rand.Restart(seed);

                // Load data
                double[,] data = LoadData(string.Format("pca-data-{0:00}.csv", seed));

                // Construct model
                int M = data.GetLength(0);
                int N = data.GetLength(1);
                Console.WriteLine("M={0} and N={1}\n", M, N);
                BayesianPCAModel bpca = new BayesianPCAModel(M, N, D);
                if (!(bpca.engine.Algorithm is VariationalMessagePassing))
                {
                    Console.WriteLine("This example only runs with Variational Message Passing");
                    return;
                }
	
                // Set the data
                bpca.data.ObservedValue = data;
                // Initialize the W marginal to break symmetry
                bpca.W.InitialiseTo(RandomGaussianVectorArray(M, D));

                double logEvidence = 0;
                var loglike = new List<double>();
                var cputime = new List<double>();
                double starttime, endtime;
                for (int j=0; j < maxiter; j++)
                {
                    starttime = get_cputime();
                    bpca.engine.NumberOfIterations = j+1;
                    // Infer the marginals
                    //bpca.engine.Infer(bpca.Alpha);
                    bpca.engine.Infer(bpca.W);
                    bpca.engine.Infer(bpca.Z);
                    bpca.engine.Infer(bpca.tau);
                    // Measure CPU time
                    endtime = get_cputime();
                    cputime.Add((endtime - starttime) / 1000.0);
                    // Compute lower bound
                    logEvidence = bpca.engine.Infer<Bernoulli>(bpca.evidence).LogOdds;
                    // Print progress
                    Console.WriteLine("Iteration {0}: loglike={1} ({2} ms)",
                                      j+1,
                                      logEvidence,
                                      endtime - starttime);
                }
                
                SaveResults(string.Format("pca-results-{0:00}-infernet.csv", seed),
                            loglike.ToArray(), cputime.ToArray());
                
                //Console.WriteLine("q(W) =\n" + bpca.engine.Infer(bpca.W));
                //Console.WriteLine("q(X) =\n" + bpca.engine.Infer(bpca.Z));
                Console.WriteLine("q(tau) =\n" + bpca.engine.Infer(bpca.tau));

            }

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
        //*/

        private static void printMatrixToConsole(Gaussian[,] matrix)
            {
                for (int i = 0; i < matrix.GetLength(0); i++)
                {
                    for (int j = 0; j < matrix.GetLength(1); j++)
                        Console.Write("{0,5:0.00}\t", matrix[i, j].GetMean());
                    Console.WriteLine("");
                }
            }

        
        public double[,] LoadData(string filename)
            {
                StreamReader sr = new StreamReader(Path.GetFullPath(filename));
                var lines = new List<double[]>();
                int Row = 0;
                while (!sr.EndOfStream)
                {
                    string[] Line = sr.ReadLine().Split(',');
                    double[] Columns = new double[Line.Length];
                    for (int i = 0; i < Line.Length; i++)
                    {
                        Columns[i] = double.Parse(Line[i]);
                    }
                    lines.Add(Columns);
                    Row++;
                }

                double[][] array = lines.ToArray();

                int M = array.Length;
                int N = array[0].Length;
                double[,] data = new double[M,N];

                for (int i = 0; i < M; i++)
                {
                    for (int j = 0; j < N; j++)
                    {
                        data[i,j] = array[i][j];
                    }
                }

                return data;

            }


        public void SaveResults(string filename, double[] loglike, double[] cputime)
            {
                using (System.IO.StreamWriter file = new System.IO.StreamWriter(Path.GetFullPath(filename)))
                {
                    for (int i = 0; i < loglike.Length; i++)
                    {
                        file.WriteLine(loglike[i] + "," + cputime[i]);
                    }
                }

            }
    }
}
