package org.onnx4j.model.graph.node;

import org.onnx4j.onnx.NamedOnnxObject;

public abstract class Attribute<T> extends NamedOnnxObject {
	
	private T value;
	
	public Attribute(T value, String name, String docString) {
		super(name, docString);
		this.value = value;
	}

	public T getValue() {
		return value;
	}

}
