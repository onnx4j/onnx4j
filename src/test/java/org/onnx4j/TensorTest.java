package org.onnx4j;

import static org.junit.Assert.assertEquals;

import org.junit.Test;
import org.onnx4j.tensor.DataType;
import org.onnx4j.tensor.Shape;

/**
 * Unit test for Tensor.class
 */
public class TensorTest {

	/**
	 * @throws Exception
	 * 
	 */
	@Test
	public void testToString() throws Exception {
		Tensor.Builder builder = Tensor.builder(DataType.FLOAT, Shape.create(2L, 3L, 3L), Tensor.options());
		for (int n = 0; n < 2 * 3 * 3; n++) {
			builder.putFloat(new Float(n));
		}

		try (Tensor ts = builder.build()) {
			String expectedOutput = "Tensor[2, 3, 3] = [[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0],],[[9.0,10.0,11.0],[12.0,13.0,14.0],[15.0,16.0,17.0],],]";
			String actualOutput = ts.toString().replaceAll("[\t\n]", "");
			assertEquals(expectedOutput, actualOutput);
			System.out.println(actualOutput);
		}
	}

}
