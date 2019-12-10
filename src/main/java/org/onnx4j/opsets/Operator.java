package org.onnx4j.opsets;

import org.onnx4j.tensor.DataType;

public interface Operator extends Forwardable {

	public enum OperatorStatus {
		EXPERIMENTAL, STABLE
	}

	public long getSinceVersion();

	public OperatorStatus getStatus();

	public String getOpType();

	public DataType[] getTypeConstraints();

	public default String getDocString() {
		return "https://github.com/onnx/onnx/blob/master/docs/Operators.md#" + getOpType();
	}

}
