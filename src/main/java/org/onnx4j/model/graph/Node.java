package org.onnx4j.model.graph;

import java.util.Arrays;

import org.onnx4j.Tensor;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.onnx.NamedOnnxObject;
import org.onnx4j.onnx.prototypes.OnnxProto3.NodeProto;

public final class Node extends NamedOnnxObject {

	protected String domain;
	protected String opType;
	protected String[] inputNames;
	protected String[] outputNames;
	protected Attributes attributes;
	
	public Node(NodeProto nodeProto, Tensor.Options tensorOptions) {
		super(nodeProto.getName(), nodeProto.getDocString());
		
		this.inputNames = nodeProto.getInputList().toArray(
				new String[nodeProto.getInputList().size()]);
		this.outputNames = nodeProto.getOutputList().toArray(
				new String[nodeProto.getOutputList().size()]);

		this.domain = nodeProto.getDomain();
		this.opType = nodeProto.getOpType();
		this.attributes = new Attributes(nodeProto.getAttributeList(), tensorOptions);
	}

	public String[] getInputNames() {
		return inputNames;
	}

	public String[] getOutputNames() {
		return outputNames;
	}

	public Attributes getAttrs() {
		return attributes;
	}
	
	public String getOpType() {
		return opType;
	}
	
	@Override
	public void close() throws Exception {
		this.attributes.close();
	}

	@Override
	public String toString() {
		return "Node [domain=" + domain + ", opType=" + opType + ", inputNames=" + Arrays.toString(inputNames)
				+ ", outputNames=" + Arrays.toString(outputNames) + ", attributes=" + attributes + "]";
	}

}
