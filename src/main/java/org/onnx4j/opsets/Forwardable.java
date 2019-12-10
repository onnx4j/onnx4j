package org.onnx4j.opsets;

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;

public interface Forwardable {

	//public OperatorOutputs forward(OperatorInputs inputs);

	//public OperatorOutputs doDiff(Node node, Inputs inputs);

	//public OperatorInputs wrapOperatorInputs(Node node, Inputs inputs);

	public Outputs forward(Node node, Inputs inputs);/* {
		//OperatorInputs operatorInputs = this.wrapOperatorInputs(node, inputs);
		return this.doDiff(node, inputs).toOutputs(node);
	}*/

}
