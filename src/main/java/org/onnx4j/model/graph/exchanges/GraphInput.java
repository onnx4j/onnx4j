package org.onnx4j.model.graph.exchanges;

import org.onnx4j.model.graph.Exchange;
import org.onnx4j.onnx.prototypes.OnnxProto3.ValueInfoProto;

public final class GraphInput extends Exchange {

	public GraphInput(ValueInfoProto valueInfoProto) {
		super(valueInfoProto);
	}

	@Override
	public void close() throws Exception {}
	
}
