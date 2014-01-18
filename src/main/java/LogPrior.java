public class LogPrior
{
	final double sigma;

	public LogPrior(final double sigma)
	{
		this.sigma = sigma;
	}

	public double compute(final double[] x, final double[] grad)
	{
		double val = 0.0;

		for (int i = 0; i < x.length; i++)
		{
			val += x[i] * x[i] / 2.0 / (sigma * sigma);
			grad[i] += x[i] / (sigma * sigma);
		}
		return val;
	}

}
