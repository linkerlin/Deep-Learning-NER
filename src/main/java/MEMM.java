import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;

/** Do not modify this class **/
public class MEMM
{

	public static void main(final String[] args) throws IOException
	{

		boolean print = false;
		boolean submit = false;

		if (args.length > 2)
		{
			if (args[2].equals("-print"))
			{
				print = true;
			}
			else if (args[2].equals("-submit"))
			{
				submit = true;
			}
		}

		final List<Datum> testData = runMEMM(args[0], args[1]);

		// print words + guess labels for development
		if (print)
		{
			for (final Datum datum : testData)
			{
				System.out.println(base64decode(datum.word) + "\t" + datum.label + "\t" + datum.guessLabel);
			}
		}

		// print guess labels for submission
		if (submit)
		{
			for (final Datum datum : testData)
			{
				System.out.println("+++" + base64decode(datum.word) + "\t" + datum.guessLabel);
			}
			return;
		}

		System.out.println();
		Scorer.score(testData);

	}

	public static List<Datum> runMEMM(final String trainFile, final String testFile) throws IOException
	{

		final List<Datum> trainData = readData(trainFile);
		final List<Datum> testDataWithMultiplePrevLabels = readData(testFile);

		final LogConditionalObjectiveFunction obj = new LogConditionalObjectiveFunction(trainData);
		final double[] initial = new double[obj.domainDimension()];

		// restore the original test data from the source
		final List<Datum> testData = new ArrayList<Datum>();
		testData.add(testDataWithMultiplePrevLabels.get(0));
		for (int i = 1; i < testDataWithMultiplePrevLabels.size(); i += obj.labelIndex.size())
		{
			testData.add(testDataWithMultiplePrevLabels.get(i));
		}

		final QNMinimizer minimizer = new QNMinimizer(15);
		final double[][] weights = obj.to2D(minimizer.minimize(obj, 1e-4, initial, -1, null));

		final Viterbi viterbi = new Viterbi(obj.labelIndex, obj.featureIndex, weights);
		viterbi.decode(testData, testDataWithMultiplePrevLabels);

		return testData;
	}

	// Read words, labels, and features
	private static List<Datum> readData(final String filename) throws IOException
	{
		final List<Datum> data = new ArrayList<Datum>();
		// read the JSON file
		final FileInputStream fstream = new FileInputStream(filename);
		JSONTokener tokener = null;

		try
		{
			tokener = new JSONTokener(fstream);
			while (tokener.more())
			{
				final JSONObject object = (JSONObject) tokener.nextValue();
				if (object == null)
				{
					break;
				}

				final String word = object.getString("_word");
				final String label = object.getString("_label");
				final String previousLabel = object.getString("_prevLabel");

				final JSONObject featureObject = (JSONObject) object.get("_features");
				final List<String> features = new ArrayList<String>();
				for (final String name : JSONObject.getNames(featureObject))
				{
					features.add(featureObject.getString(name));
				}

				final Datum datum = new Datum(word, label);
				datum.features = features;
				datum.previousLabel = previousLabel;

				data.add(datum);
			}
		}
		catch (final JSONException e)
		{
			System.err.println("BEFORE " + data.size() + " " + data.get(data.size() - 1));
			e.printStackTrace();
		}

		return data;
	}

	private static String base64decode(final String str)
	{
		final Base64 base = new Base64();
		final byte[] strBytes = str.getBytes();
		final byte[] decodedBytes = base.decode(strBytes);
		final String decoded = new String(decodedBytes);
		return decoded;
	}
}