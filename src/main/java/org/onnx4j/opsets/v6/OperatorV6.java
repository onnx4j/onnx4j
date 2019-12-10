package org.onnx4j.opsets.v6;

import org.onnx4j.opsets.v5.OperatorV5;

public interface OperatorV6 extends OperatorV5 {

	public default long getSinceVersion() {
		return 6L;
	}

}
