package org.onnx4j.opsets.v4;

import org.onnx4j.opsets.v3.OperatorV3;

public interface OperatorV4 extends OperatorV3 {

	public default long getSinceVersion() {
		return 4L;
	}

}
