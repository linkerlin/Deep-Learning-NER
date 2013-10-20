import java.util.HashMap;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class WindowModel
{

	// protected static int QN_MIN_MEM= 15;
	// protected Random rgen = new Random();
	// protected GradientChecker checker = new GradientChecker();

	// //---------------MAIN REQUIRED PARAMETERS------------------
	// protected int wordSize; //stores model parameters
	// protected HashMap<String, Integer> wordMap; //map for global word IDs

	protected SimpleMatrix Wv, W, Wout;
	public int windowSize, wordSize, hiddenSize;

	public HashMap<String, Integer> wordToNum;
	public HashMap<Integer, String> numToWord;
	List<Datum> trainData;

	// learning rate
	double lr;

	protected int maxIter, miniBatchSize;
	protected double funcTol, regC, regC_Wv, regC_Wout;
	protected boolean trainAllParams, doGradientCheck;
	protected int optimizationMethod;

	protected int currentOptimizerIter;
	protected Random rgen = new Random();

	/**
	 * Constructor: Initialize the weights randomly
	 */
	public WindowModel(final FeatureFactory ff, final int _windowSize, final int _hiddenSize, final double _lr)
	{
		Wv = ff.allVecs;
		wordToNum = ff.wordToNum;
		numToWord = ff.numToWord;
		wordSize = Wv.numRows();

		// learning rate
		lr = _lr;
		windowSize = _windowSize;
		hiddenSize = _hiddenSize;
		initWeights();
	}

	/**
	 * Initializes the weights randomly. Some using identity.
	 */
	public void initWeights()
	{

		final int fanIn = wordSize * windowSize;
		// initialize with bias inside as the last column
		W = SimpleMatrix.random(hiddenSize, fanIn + 1, -1 / Math.sqrt(fanIn), 1 / Math.sqrt(fanIn), rgen);

		// random vector
		Wout = SimpleMatrix.random(1, hiddenSize, -1 / Math.sqrt(fanIn), 1 / Math.sqrt(fanIn), rgen);
	}

	/**
	 * Simplest SGD training possible
	 */
	public void train(final List<Datum> _trainData)
	{
		trainData = _trainData;
		final int totalIter = 1;
		final int numWordsInTrain = trainData.size();
		for (int iter = 0; iter < totalIter; iter++)
		{

			for (int i = 0; i < numWordsInTrain; i++)
			{
				// for (int i = 0;i<100;i++){
				final Datum datum = trainData.get(i);
				int y;
				if (datum.label.equals("O"))
				{
					y = 0;
				}
				else
				{
					y = 1;
				}

				// forward prop
				final int[] windowNums = getWindowNumsTrain(i);
				final SimpleMatrix allX = getWindowVectorWithBias(windowNums);
				final SimpleMatrix h = tanh(W.mult(allX));
				final double p_pred = sigmoid(Wout.mult(h).get(0));

				// compute derivatives
				final SimpleMatrix Wout_df = h.scale(y - p_pred);
				final SimpleMatrix allXT = allX.transpose();
				final SimpleMatrix W_df = tanhDer(h).scale(y - p_pred).mult(allXT);
				// TODO: Update word vectors

				// update with simple SGD step
				Wout = Wout.plus(Wout_df.scale(lr).transpose());
				W = W.plus(W_df.scale(lr));

				// // check if prob is higher?
				// allX = getWindowVectorWithBias(windowNums);
				// h = tanh(W.mult(allX));
				// double p_predNew = sigmoid((double)Wout.mult(h).get(0));
				// System.out.println("Label:"+Integer.toString(y)+" Old: "+Double.toString(p_pred)+", new: "+Double.toString(p_predNew));
				// System.out.println(".");
			}

			// test on train set
			// int tp=0;
			// int tn=0;
			// int fp=0;
			// int fn=0;
			// for (int i = 0;i<numWordsInTrain;i++){
			// // for (int i = 0;i<100;i++){
			// Datum datum = trainData.get(i);
			// int y;
			// if (datum.label.equals("O")){y = 0;} else {y= 1;}
			// // forward prop
			// int[] windowNums = getWindowNumsTrain(i);
			// SimpleMatrix allX = getWindowVectorWithBias(windowNums);
			// SimpleMatrix h = tanh(W.mult(allX));
			// double p_pred = sigmoid((double)Wout.mult(h).get(0));
			// if (p_pred>0.5 && y==1){
			// tp++;
			// }else if (p_pred>0.5 && y==0) {
			// fp++;
			// }else if (p_pred<=0.5 && y==0) {
			// tn++;
			// }else if (p_pred<=0.5 && y==1) {
			// fn++;
			// }
			// }
			// double prec = (double)tp/(tp+fp);
			// double rec = (double)tp/(tp+fn);
			// double f1 = (double)2.0*prec*rec/(prec+rec);
			// System.out.println("Training Precision="+Double.toString(prec)+", Recall="+Double.toString(rec)+", F1="+Double.toString(f1));
		}
	}

	public void test(final List<Datum> testData)
	{
		int tp = 0;
		int tn = 0;
		int fp = 0;
		int fn = 0;
		final int numWordsInTrain = testData.size();

		for (int i = 0; i < numWordsInTrain; i++)
		{
			final Datum datum = testData.get(i);
			int y;
			if (datum.label.equals("O"))
			{
				y = 0;
			}
			else
			{
				y = 1;
			}
			// forward prop
			final int[] windowNums = getWindowNumsTest(i, testData);
			final SimpleMatrix allX = getWindowVectorWithBias(windowNums);
			final SimpleMatrix h = tanh(W.mult(allX));
			final double p_pred = sigmoid(Wout.mult(h).get(0));
			if (p_pred > 0.5 && y == 1)
			{
				tp++;
			}
			else if (p_pred > 0.5 && y == 0)
			{
				fp++;
			}
			else if (p_pred <= 0.5 && y == 0)
			{
				tn++;
			}
			else if (p_pred <= 0.5 && y == 1)
			{
				fn++;
			}
		}
		final double prec = (double) tp / (tp + fp);
		final double rec = (double) tp / (tp + fn);
		final double f1 = 2.0 * prec * rec / (prec + rec);
		System.out.println("Test: Precision=" + Double.toString(prec) + ", Recall=" + Double.toString(rec) + ", F1=" + Double.toString(f1));
	}

	private SimpleMatrix getWindowVectorWithBias(final int[] windowNums)
	{
		final SimpleMatrix allX = new SimpleMatrix(wordSize * windowSize + 1, 1);
		for (int i = 0; i < windowSize; i++)
		{
			allX.insertIntoThis(i * wordSize, 0, Wv.extractVector(false, windowNums[i]));
		}
		// adding bias
		allX.set(allX.numRows() - 1, 0, 1);
		return allX;
	}

	private int[] getWindowNumsTest(final int wordPos, final List<Datum> testData)
	{
		final int[] windowNums = new int[windowSize];
		final int startSymbol = wordToNum.get("<s>");
		final int endSymbol = wordToNum.get("</s>");
		final int contextSize = (int) Math.floor((windowSize - 1) / 2);
		int counter = 0;
		for (int i = wordPos - contextSize; i <= wordPos + contextSize; i++)
		{
			if (i < 0)
			{
				windowNums[counter] = startSymbol;
			}
			else if (i > testData.size())
			{
				windowNums[counter] = endSymbol;
			}
			else
			{
				windowNums[counter] = getWordIDTest(i, testData);
			}
			counter++;
		}

		return windowNums;
	}

	private int[] getWindowNumsTrain(final int wordPos)
	{
		final int[] windowNums = new int[windowSize];
		final int startSymbol = wordToNum.get("<s>");
		final int endSymbol = wordToNum.get("</s>");
		final int contextSize = (int) Math.floor((windowSize - 1) / 2);
		int counter = 0;
		for (int i = wordPos - contextSize; i <= wordPos + contextSize; i++)
		{
			if (i < 0)
			{
				windowNums[counter] = startSymbol;
			}
			else if (i > trainData.size())
			{
				windowNums[counter] = endSymbol;
			}
			else
			{
				windowNums[counter] = getWordIDTrain(i);
			}
			counter++;
		}

		return windowNums;
	}

	public int getWordIDTest(final int position, final List<Datum> testData)
	{
		int out;
		try
		{
			out = wordToNum.get(testData.get(position).word);
		}
		catch (final Exception e)
		{
			// UNK=0
			out = 0;
		}
		return out;
	}

	public int getWordIDTrain(final int position)
	{
		int out;
		try
		{
			out = wordToNum.get(trainData.get(position).word);
		}
		catch (final Exception e)
		{
			// UNK=0
			out = 0;
		}
		return out;
	}

	/**
	 * Performs element-wise tanh function.
	 */
	public SimpleMatrix tanh(final SimpleMatrix in)
	{
		final SimpleMatrix out = new SimpleMatrix(in.numRows(), in.numCols());
		for (int j = 0; j < in.numCols(); j++)
			for (int i = 0; i < in.numRows(); i++)
				out.set(i, j, Math.tanh(in.get(i, j)));
		return out;
	}

	/**
	 * Performs derivative function.
	 */
	public SimpleMatrix tanhDer(final SimpleMatrix in)
	{
		final SimpleMatrix out = new SimpleMatrix(in.numRows(), in.numCols());
		out.set(1);
		out.set(out.minus(in.elementMult(in)));
		return out;
	}

	/**
	 * Performs element-wise sigmoid function.
	 */
	public SimpleMatrix sigmoid(final SimpleMatrix in)
	{
		final SimpleMatrix out = new SimpleMatrix(in.numRows(), in.numCols());
		for (int j = 0; j < in.numCols(); j++)
			for (int i = 0; i < in.numRows(); i++)
				out.set(i, j, sigmoid(in.get(i, j)));
		return out;
	}

	/**
	 * Performs element-wise sigmoid function.
	 */
	public SimpleMatrix sigmoidDer(final SimpleMatrix in)
	{
		final SimpleMatrix ones = new SimpleMatrix(in.numRows(), in.numCols());
		ones.set(1);
		return in.elementMult(ones.minus(in));
	}

	public static double sigmoid(final double x)
	{
		return 1 / (1 + Math.exp(-x));
	}

	/**
	 * Performs element-wise tanh function. Fills the new array with these values.
	 */
	public static void elemTanh(final SimpleMatrix in, final SimpleMatrix out)
	{
		for (int j = 0; j < in.numCols(); j++)
			for (int i = 0; i < in.numRows(); i++)
				out.set(i, j, Math.tanh(in.get(i, j)));
	}

}
