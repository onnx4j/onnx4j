package org.onnx4j.opsets.v1;

import org.onnx4j.opsets.Operator;

public interface OperatorV1 extends Operator {

	public default long getSinceVersion() {
		return 1L;
	}

}
