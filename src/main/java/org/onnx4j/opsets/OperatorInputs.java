package org.onnx4j.opsets;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;

public abstract class OperatorInputs {

	private Node node;
	private Inputs inputs;

	public OperatorInputs(Node node, Inputs inputs) {
		this.node = node;
		this.inputs = inputs;
	}
	
	public <T extends OperatorInputs> T cast(Class<T> clazz) {
	    return clazz.isInstance(this) ? clazz.cast(this) : null;
	}

	protected Node getNode() {
		return node;
	}

	protected Inputs getInputs() {
		return inputs;
	}

}
