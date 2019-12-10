package org.onnx4j.onnx;

public abstract class OnnxObject implements AutoCloseable {

	protected String docString;

	public OnnxObject(String docString) {
		this.docString = docString;
	}

	public String getDocString() {
		return this.docString;
	}

}
