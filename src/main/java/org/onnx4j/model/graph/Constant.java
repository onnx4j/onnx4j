package org.onnx4j.model.graph;

import org.onnx4j.Tensor;
import org.onnx4j.onnx.NamedOnnxObject;
import org.onnx4j.onnx.prototypes.OnnxProto3.TensorProto;

public class Constant extends NamedOnnxObject {

	private Tensor tensor;

	public Constant(TensorProto initializer, Tensor.Options tensorOptions) {
		super(initializer.getName(), initializer.getDocString());
		this.tensor = Tensor.toTensor(initializer, tensorOptions);
	}

	public Tensor getTensor() {
		return tensor;
	}

	@Override
	public void close() throws Exception {
		if (this.tensor != null) {
			this.tensor.close();
			this.tensor = null;
		}
	}

}
