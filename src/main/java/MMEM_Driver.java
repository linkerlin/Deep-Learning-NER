import java.io.IOException;

public class MMEM_Driver
{

	public static void main(final String[] args) throws IOException
	{
		MEMM.main(new String[] { System.getProperty("user.dir") + "/src/test/resources/" + "trainWithFeatures.json",
				System.getProperty("user.dir") + "/src/test/resources/" + "testWithFeatures.json", "-print" });
	}

}
