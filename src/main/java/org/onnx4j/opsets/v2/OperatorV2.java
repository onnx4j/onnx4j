package org.onnx4j.opsets.v2;

import org.onnx4j.opsets.v1.OperatorV1;

public interface OperatorV2 extends OperatorV1 {

	public default long getSinceVersion() {
		return 2L;
	}

}
