package org.onnx4j.opsets;

import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;

public abstract class OperatorOutputs {

	public abstract Outputs toOutputs(Node node);
	
}
