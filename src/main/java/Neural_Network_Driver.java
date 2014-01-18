import java.io.IOException;

import static java.lang.System.getProperty;

public class Neural_Network_Driver
{
	public static void main(final String[] args) throws IOException
	{
		NER.main(new String[] { getProperty("user.dir") + "/src/main/resources/" + "train",
				getProperty("user.dir") + "/src/main/resources/" + "dev" });
	}

}
