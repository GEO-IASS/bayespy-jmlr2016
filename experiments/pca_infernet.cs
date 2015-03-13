// Copyright (c) 2015 Jaakko Luttinen
// MIT License

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

    public interface BayesianPCAModel
    {
        void create(int M, int N, int D);

        void observe(double[,] obs);

        void update();

        double bound();

        void print();
        
    }


    public class BayesianPCAModel_Factorized : BayesianPCAModel
    {
        // Inference engine
        public InferenceEngine engine = null;
        // Model variables
        public VariableArray2D<double> data = null;
        public VariableArray2D<double> W = null;
        public VariableArray2D<double> Z = null;
        public Variable<double> tau = null;
        public VariableArray<double> Alpha = null;
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

        public void observe(double[,] obs)
            {
                data.ObservedValue = obs;
            }

        public void update()
            {
                engine.NumberOfIterations++;
                engine.Infer(Alpha);
                engine.Infer(W);
                engine.Infer(Z);
                engine.Infer(tau);
            }

        public double bound()
            {
                return engine.Infer<Bernoulli>(evidence).LogOdds;                
            }

        public void print()
            {
                Console.WriteLine("q(tau) =\n" + engine.Infer(tau));
            }
        
        public void create(int M, int N, int D)
            {

                // Stuff for the lower bound / evidence
                IfBlock block = Variable.If(evidence);

                rN = new Range(N).Named("N");
                rD = new Range(D).Named("D");
                rM = new Range(M).Named("M");
                
                // Mixing matrix
                Alpha = Variable.Array<double>(rD).Named("Alpha");
                Alpha[rD] = Variable.GammaFromShapeAndRate(0.01, 0.01).ForEach(rD);
                W = Variable.Array<double>(rM, rD).Named("W");
                W[rM, rD] = Variable.GaussianFromMeanAndPrecision(0, Alpha[rD]).ForEach(rM);

                // Latent variables
                Z = Variable.Array<double>(rD, rN).Named("Z");
                Z[rD, rN] = Variable.GaussianFromMeanAndPrecision(0.0, 1.0).ForEach(rD, rN);

                // Multiply
                var ZtimesW = Variable.MatrixMultiply(W, Z).Named("Z*W");

                // Observation noise
                tau = Variable.GammaFromShapeAndRate(0.01, 0.01);

                // Observations
                data = Variable.Array<double>(rM, rN).Named("Y");
                data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZtimesW[rM, rN], tau);

                // End evidence/lowerbound block
                block.CloseBlock();
                        
                // Inference engine
                engine = new InferenceEngine(new VariationalMessagePassing());

		W.InitialiseTo(randomGaussianArray(M, D));
                engine.NumberOfIterations = 0;
                        

                return;
            }
        
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
    }

    public class BayesianPCAModel_Full : BayesianPCAModel
    {
        // Inference engine
        public InferenceEngine engine = null;
        // Model variables
        public Variable<PositiveDefiniteMatrix> Alpha = null;
        public VariableArray<Vector> W = null;
        public VariableArray<Vector> Z = null;
        //public VariableArray<double> mu = null;
        public VariableArray2D<double> data = null;
        public Variable<double> tau = null;
        public Range rN = null;
        public Range rD = null;
        public Range rM = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = Variable.Bernoulli(0.5).Named("evidence");

        public void observe(double[,] obs)
            {
                data.ObservedValue = obs;
            }

        public void update()
            {
                engine.NumberOfIterations++;
                engine.Infer(Alpha);
                engine.Infer(W);
                engine.Infer(Z);
                engine.Infer(tau);
            }

        public double bound()
            {
                return engine.Infer<Bernoulli>(evidence).LogOdds;                
            }

        public void print()
            {
                Console.WriteLine("q(tau) =\n" + engine.Infer(tau));
            }
        
        public void create(int M, int N, int D)
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
                W[rM] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                    Alpha).ForEach(rM);

                // Latent variables
                Z = Variable.Array<Vector>(rN).Named("Z");
                Z[rN] = Variable.VectorGaussianFromMeanAndPrecision(Vector.Zero(D),
                                                                    PositiveDefiniteMatrix.Identity(D)).ForEach(rN);

                // Multiply
                var ZtimesW = Variable.Array<double>(rM, rN).Named("ZtimesW");
                ZtimesW[rM, rN] = Variable.InnerProduct(W[rM], Z[rN]);

                tau = Variable.GammaFromShapeAndRate(0.01, 0.01);

                // Observations
                data = Variable.Array<double>(rM, rN).Named("Y");
                data[rM, rN] = Variable.GaussianFromMeanAndPrecision(ZtimesW[rM,rN], tau);

                // End evidence/lowerbound block
                block.CloseBlock();
                        
                // Inference engine
                engine = new InferenceEngine(new VariationalMessagePassing());

                // Initialize the W marginal to break symmetry
                W.InitialiseTo(RandomGaussianVectorArray(M, D));
                engine.NumberOfIterations = 0;

                return;
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
    }

    public class Program
    {
        
        static void Main(string[] args)
            {
                    
                Console.Write("Running PCA.\n");
                var bpca = new BayesianPCA();
                bool factorize = int.Parse(args[3]) != 0;
                bpca.Run(int.Parse(args[0]),  // components
                         int.Parse(args[1]),  // seed
                         int.Parse(args[2]),  // maxiter
                         factorize);
                return;
            }

    }

    
    public class BayesianPCA
    {

        public void Run(int D, int seed, int maxiter, bool factorize)
            {
                
                // Restart the infer.NET random number generator
                Rand.Restart(seed);

                // Load data
                double[,] data = LoadData(string.Format("pca-data-{0:00}.csv", seed));

                // Construct model
                int M = data.GetLength(0);
                int N = data.GetLength(1);
                Console.WriteLine("M={0} and N={1}\n", M, N);
                BayesianPCAModel bpca;
                if (factorize)
                {
                    bpca = new BayesianPCAModel_Factorized();
                }
                else
                {
                    bpca = new BayesianPCAModel_Full();
                }
                bpca.create(M, N, D);
	
                // Set the data
                bpca.observe(data); //data.ObservedValue = data;

                double logEvidence = 0;
                var loglike = new List<double>();
                var cputime = new List<double>();
                double starttime, endtime;
                for (int j=0; j < maxiter; j++)
                {
                    starttime = get_cputime();
                    bpca.update();
                    // Measure CPU time
                    endtime = get_cputime();
                    cputime.Add((endtime - starttime) / 1000.0);
                    // Compute lower bound
                    logEvidence = bpca.bound(); //bpca.engine.Infer<Bernoulli>(bpca.evidence).LogOdds;
                    loglike.Add(logEvidence);
                    // Print progress
                    Console.WriteLine("Iteration {0}: loglike={1} ({2} ms)",
                                      j+1,
                                      logEvidence,
                                      endtime - starttime);
                }
                
                SaveResults(string.Format("pca-results-{0:00}-infernet.csv", seed),
                            loglike.ToArray(), cputime.ToArray());
                
                bpca.print();

            }

        public double get_cputime()
            {
                Process process = Process.GetCurrentProcess();
                return process.TotalProcessorTime.TotalMilliseconds;
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
