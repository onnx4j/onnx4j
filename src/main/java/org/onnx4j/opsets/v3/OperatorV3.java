package org.onnx4j.opsets.v3;

import org.onnx4j.opsets.v2.OperatorV2;

public interface OperatorV3 extends OperatorV2 {

	public default long getSinceVersion() {
		return 3L;
	}

}
