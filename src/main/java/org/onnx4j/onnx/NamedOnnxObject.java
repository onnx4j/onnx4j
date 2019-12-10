package org.onnx4j.onnx;

import org.onnx4j.model.graph.Node;

public abstract class NamedOnnxObject extends OnnxObject {

	protected String name;

	public NamedOnnxObject(String name, String docString) {
		super(docString);
		this.name = name;
	}

	public String getName() {
		return this.name;
	}

	@Override
	public boolean equals(Object object) {
		if (Node.class.isInstance(object) == false)
			return false;

		return name.equalsIgnoreCase(((NamedOnnxObject) object).name);
	}

}
