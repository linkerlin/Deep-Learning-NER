import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import static java.lang.Double.parseDouble;

import org.ejml.simple.SimpleMatrix;

public class FeatureFactory
{
	static class Data
	{
		public Data(Map[] vocabulary, SimpleMatrix vectors_from)
		{
			wordToNum = vocabulary[0];
			numToWord = vocabulary[1];
			allVecs = vectors_from;
		}

		Map<String, Integer> wordToNum;
		Map<Integer, String> numToWord;
		SimpleMatrix allVecs;
	}

	public List<Datum> readData(String filename) throws IOException
	{
		List<Datum> data = new ArrayList<>();

		try (Scanner scanner = new Scanner(new File(filename)))
		{
			while (scanner.hasNextLine())
			{
				String nextLine = scanner.nextLine().trim();
				if (nextLine.isEmpty())
					continue;
				String[] bits = nextLine.split("\\s+");
				String word = bits[0];
				String label = bits[1];
				data.add(new Datum(word, label));
			}
		}

		return data;
	}

	public Data readWordVectors(String vecFilename, String vocabFilename) throws IOException
	{
		Map[] vocabulary = vocabulary_from(vocabFilename);

		return new Data(vocabulary, vectors_from(vecFilename, vocabulary[0].size()));
	}

	private Map[] vocabulary_from(String vocabFilename) throws FileNotFoundException
	{
		int counter = 0;

		Map<String, Integer> wordToNum = new HashMap<>();
		Map<Integer, String> numToWord = new HashMap<>();

		try (Scanner scanner = new Scanner(new File(vocabFilename)))
		{
			while (scanner.hasNextLine())
			{
				String nextLine = scanner.nextLine().trim();
				if (nextLine.isEmpty())
					continue;
				String[] bits = nextLine.split("\\s+");
				String word = bits[0];
				wordToNum.put(word, counter);
				numToWord.put(counter, word);
				counter++;
			}
		}
		return new Map[] { wordToNum, numToWord };
	}

	SimpleMatrix vectors_from(String vector_file, int columns) throws FileNotFoundException
	{
		int VECTOR_SIZE = 50;
		SimpleMatrix allVecs = new SimpleMatrix(VECTOR_SIZE, columns);
		int counter = 0;

		try (Scanner scanner = new Scanner(new File(vector_file)))
		{
			while (scanner.hasNextLine())
			{
				String nextLine = scanner.nextLine().trim();

				if (nextLine.isEmpty())
					continue;

				String[] bits = nextLine.split("\\s+");
				for (int pos = 0; pos < VECTOR_SIZE; pos++)
				{
					allVecs.set(pos, counter, parseDouble(bits[pos]));
				}

				counter++;
			}
		}

		return allVecs;
	}

}
