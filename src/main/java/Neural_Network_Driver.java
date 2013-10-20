import java.io.IOException;

public class Neural_Network_Driver
{
	public static void main(final String[] args) throws IOException
	{
		NER.main(new String[] { System.getProperty("user.dir") + "/src/main/resources/" + "train",
				System.getProperty("user.dir") + "/src/main/resources/" + "dev" });
	}

}
