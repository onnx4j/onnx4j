package org.onnx4j.model.graph.exchanges;

import org.onnx4j.model.graph.Exchange;
import org.onnx4j.model.graph.Node;
import org.onnx4j.onnx.prototypes.OnnxProto3.ValueInfoProto;

public final class GraphOutput extends Exchange {
	
	private Node node;

	public GraphOutput(Node node, ValueInfoProto valueInfoProto) {
		super(valueInfoProto);
		this.node = node;
	}
	
	public Node getNode() {
		return this.node;
	}
	
	@Override
	public void close() throws Exception {}
	
}
