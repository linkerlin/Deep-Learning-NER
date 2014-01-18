import static org.hamcrest.Matchers.is;
import static org.junit.Assert.*;
import static java.lang.System.getProperty;

import org.junit.Test;

public class Neural_Network_Integration_Test
{
	@Test public void gives_expected_precision_recall_and_f1() throws Exception
	{
		String result = NER.train_and_test(new String[] { getProperty("user.dir") + "/src/main/resources/" + "train",
				getProperty("user.dir") + "/src/main/resources/" + "dev" });

		assertThat(result, is("Test: Precision=0.8021806853582555, Recall=0.4906319466497301, F1=0.6088669950738916"));
	}
}
