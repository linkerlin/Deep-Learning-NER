import java.util.List;

public class Datum
{

	@Override public String toString()
	{
		return "Datum [word=" + word + ", label=" + label + ", features=" + features + ", guessLabel=" + guessLabel + ", previousLabel="
				+ previousLabel + "]";
	}

	public final String word;
	public final String label;
	public List<String> features;
	public String guessLabel;
	public String previousLabel;

	public Datum(final String word, final String label)
	{
		this.word = word;
		this.label = label;
	}
}