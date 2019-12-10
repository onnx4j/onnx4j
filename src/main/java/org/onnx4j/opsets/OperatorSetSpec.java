package org.onnx4j.opsets;

import java.util.HashMap;
import java.util.Map;

public interface OperatorSetSpec {

	public default Map<String, Operator> initializeOperators() {
		return new HashMap<String, Operator>();
	}

}
