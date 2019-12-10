package org.onnx4j.opsets.v5;

import org.onnx4j.opsets.v4.OperatorV4;

public interface OperatorV5 extends OperatorV4 {

	public default long getSinceVersion() {
		return 5L;
	}

}
