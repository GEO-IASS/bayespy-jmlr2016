using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Text;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;

namespace MicrosoftResearch.Infer.Tutorials
{

    public class MixtureOfGaussiansModel
    {
        public VariableArray<Vector> means = null;
        public VariableArray<PositiveDefiniteMatrix> precs = null;
        public Variable<Vector> weights = null;
        public VariableArray<Vector> data = null;
        public VariableArray<int> z = null;

        public Variable<int> K = null;
        public Variable<int> N = null;
        public Variable<int> D = null;
        public Range k = null;
        public Range n = null;
        public Range d = null;

        public InferenceEngine engine = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = null;

        public MixtureOfGaussiansModel(int N, int K, int D)
            {
                // Stuff for the lower bound / evidence
                evidence = Variable.Bernoulli(0.5).Named("evidence");
                IfBlock block = Variable.If(evidence);

                // Define a range for the number of mixture components
                //K = Variable.New<int>().Named("NumComps");
                //N = Variable.New<int>().Named("NumObs");
                //D = Variable.New<int>().Named("NumDims");
                k = new Range(K).Named("k");
                n = new Range(N).Named("n");
                d = new Range(D).Named("d");

                // Mixture component means
                means = Variable.Array<Vector>(k).Named("means");			
                means[k] = Variable.VectorGaussianFromMeanAndPrecision(
                    Vector.Zero(d.SizeAsInt),
                    PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt,0.01)).ForEach(k);
	
                // Mixture component precisions
                precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
                precs[k] = Variable.WishartFromShapeAndRate(d.SizeAsInt,
                                                            PositiveDefiniteMatrix.IdentityScaledBy(d.SizeAsInt,d.SizeAsInt)).ForEach(k);
			
                // Mixture weights
                double[] a = new double[K];
                for (int i = 0; i < K; i++)
                {
                    a[i] = 1;
                }
                weights = Variable.Dirichlet(k, a).Named("weights");	

                // Create a variable array which will hold the data
                data = Variable.Array<Vector>(n).Named("x");
                // Create latent indicator variable for each data point
                z = Variable.Array<int>(n).Named("z");

                // The mixture of Gaussians model
                using (Variable.ForEach(n)) {
                    z[n] = Variable.Discrete(weights);
                    using (Variable.Switch(z[n])) {
                        data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]]);
                    }
                }

                // End evidence/lowerbound block
                block.CloseBlock();
                    
                engine = new InferenceEngine(new VariationalMessagePassing());
            }
            
        
    }

    public class MixtureOfGaussians
    {
        static void Main(string[] args)
            {
                Console.Write("Running mixture of Gaussians\n");
                MixtureOfGaussians mog = new MixtureOfGaussians();
                mog.Run(int.Parse(args[0]), // clusters
                        int.Parse(args[1]), // seed
                        int.Parse(args[2])); // maxiter
                return;
            }

        public double get_cputime()
            {
                Process process = Process.GetCurrentProcess();
                return process.TotalProcessorTime.TotalMilliseconds;
            }
            
        public void Run(int K, int seed, int maxiter)
            {
                Vector[] data = LoadData(string.Format("mog-data-{0:00}.csv", seed));
                //data = GenerateData(300);
                
                // Restart the infer.NET random number generator
                Rand.Restart(seed);
          
                // Model size
                //int maxiter = 200;
                //int K = 40;  // clusters
                int N = data.Length; // samples
                int D = data[0].Count; // dimensionality

                MixtureOfGaussiansModel mog = new MixtureOfGaussiansModel(N, K, D);

                // Attach some generated data
                mog.data.ObservedValue = data;
                
                // Initialise messages randomly so as to break symmetry
                Discrete[] zinit = new Discrete[N];		
                for (int i = 0; i < zinit.Length; i++) 
                    zinit[i] = Discrete.PointMass(Rand.Int(K), 
                                                  K);
                mog.z.InitialiseTo(Distribution<int>.Array(zinit));

                // The inference
                //Process proc = Process.GetCurrentProcess();
                double logEvidence = 0;
                var loglike = new List<double>();
                var cputime = new List<double>();
                double starttime, endtime;
                for (int j=0; j < maxiter; j++)
                {
                    // Measure CPU time
                    starttime = get_cputime();
                    // Take one iteration more
                    mog.engine.NumberOfIterations = j+1;
                    // Infer the marginals
                    mog.engine.Infer(mog.means);
                    mog.engine.Infer(mog.precs);
                    mog.engine.Infer(mog.z);
                    mog.engine.Infer(mog.weights);
                    // Measure CPU time
                    endtime = get_cputime();
                    cputime.Add((endtime - starttime)/1000.0);
                    // Compute lower bound
                    logEvidence = mog.engine.Infer<Bernoulli>(mog.evidence).LogOdds;
                    loglike.Add(logEvidence);
                    // Print progression
                    Console.WriteLine("Iteration {0}: loglike={1} ({2} ms)",
                                      j+1,
                                      logEvidence,
                                      endtime - starttime);
                }

                Console.WriteLine("Dist over pi=" + mog.engine.Infer(mog.weights));
                //Console.WriteLine("Dist over means=\n" + mog.engine.Infer(mog.means));
                //Console.WriteLine("Dist over precs=\n" + mog.engine.Infer(mog.precs));

                SaveResults(string.Format("mog-results-{0:00}-infernet.csv", seed),
                            loglike.ToArray(), cputime.ToArray());

            }


        public Vector[] LoadData(string filename)
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
                    //Console.WriteLine(Row);
                }

                var array = lines.ToArray();
                
                Vector[] data = new Vector[array.Length];
                for (int i = 0; i < array.Length; i++)
                {
                    data[i] = Vector.FromArray(array[i]);
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
