package org.onnx4j.opsets.v7;

import org.onnx4j.opsets.v6.OperatorV6;

public interface OperatorV7 extends OperatorV6 {

	public default long getSinceVersion() {
		return 7L;
	}

}
