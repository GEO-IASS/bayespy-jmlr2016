using System;
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
        public Range k = null;
        public Range n = null;

        public InferenceEngine ie = null;

        // Variable for computing the lower bound
        public Variable<bool> evidence = null;

        public MixtureOfGaussiansModel()
            {
                // Stuff for the lower bound / evidence
                evidence = Variable.Bernoulli(0.5).Named("evidence");
                //IfBlock block = Variable.If(evidence);

                // Define a range for the number of mixture components
                K = Variable.New<int>().Named("NumComps");
                N = Variable.New<int>().Named("NumObs");
                k = new Range(K).Named("k");
                n = new Range(N).Named("n");

                // Mixture component means
                means = Variable.Array<Vector>(k).Named("means");			
                means[k] = Variable.VectorGaussianFromMeanAndPrecision(
                    Vector.FromArray(0.0,0.0),
                    PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
	
                // Mixture component precisions
                precs = Variable.Array<PositiveDefiniteMatrix>(k).Named("precs");
                precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k);
			
                // Mixture weights 
                weights = Variable.Dirichlet(k, new double[] { 1, 1 }).Named("weights");	

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
                //block.CloseBlock();
                    
                ie = new InferenceEngine(new VariationalMessagePassing());
            }
            
        
    }

    public class MixtureOfGaussians
    {
        static void Main()
            {
                Console.Write("Running mixture of Gaussians\n");
                MixtureOfGaussians mog = new MixtureOfGaussians();
                mog.Run();
                return;
            }

        public void Run()
            {
                // Start model
                MixtureOfGaussiansModel mog = new MixtureOfGaussiansModel();

                // Attach some generated data
                mog.K.ObservedValue = 2;
                mog.N.ObservedValue = 300;
                mog.data.ObservedValue = GenerateData(mog.N.ObservedValue);

                // Initialise messages randomly so as to break symmetry
                Discrete[] zinit = new Discrete[mog.N.ObservedValue];		
                for (int i = 0; i < zinit.Length; i++) 
                    zinit[i] = Discrete.PointMass(Rand.Int(mog.K.ObservedValue), 
                                                  mog.K.ObservedValue);
                mog.z.InitialiseTo(Distribution<int>.Array(zinit)); 

                // The inference
                double logEvidence = 0;
                for (int j=0; j < 100; j++)
                {
                    mog.ie.NumberOfIterations = j+1;
                    // Infer the marginals
                    mog.ie.Infer(mog.weights);
                    mog.ie.Infer(mog.means);
                    mog.ie.Infer(mog.precs);
                    // Show lower bound
                    logEvidence = mog.ie.Infer<Bernoulli>(mog.evidence).LogOdds;
                    Console.WriteLine("The lower bound: {0}", logEvidence);
                }

                Console.WriteLine("Dist over pi=" + mog.ie.Infer(mog.weights));
                Console.WriteLine("Dist over means=\n" + mog.ie.Infer(mog.means));
                Console.WriteLine("Dist over precs=\n" + mog.ie.Infer(mog.precs));

            }


        /// <summary>
        /// Generates a data set from a particular true model.
        /// </summary>
        public Vector[] GenerateData(int nData)
            {
                Vector trueM1 = Vector.FromArray(2.0, 3.0);
                Vector trueM2 = Vector.FromArray(7.0, 5.0);
                PositiveDefiniteMatrix trueP1 = new PositiveDefiniteMatrix(
                    new double[,] { { 3.0, 0.2 }, { 0.2, 2.0 } });
                PositiveDefiniteMatrix trueP2 = new PositiveDefiniteMatrix(
                    new double[,] { { 2.0, 0.4 }, { 0.4, 4.0 } });
                VectorGaussian trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
                VectorGaussian trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);
                double truePi = 0.6;
                Bernoulli trueB = new Bernoulli(truePi);
                // Restart the infer.NET random number generator
                Rand.Restart(12347);
                Vector[] data = new Vector[nData];
                for (int j = 0; j < nData; j++) {
                    bool bSamp = trueB.Sample();
                    data[j] = bSamp ? trueVG1.Sample() : trueVG2.Sample();
                }
                return data;
            }
    }
}
