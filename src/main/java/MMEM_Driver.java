import java.io.IOException;

import static java.lang.System.getProperty;

public class MMEM_Driver
{

	public static void main(final String[] args) throws IOException
	{
		MEMM.main(new String[] {  getProperty("user.dir") + "/src/test/resources/" + "trainWithFeatures.json",
				 getProperty("user.dir") + "/src/test/resources/" + "testWithFeatures.json", "-print" });
	}

}
