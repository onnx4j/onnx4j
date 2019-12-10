package org.onnx4j.opsets.v4.ops;

import org.onnx4j.opsets.v1.ops.ConcatV1;
import org.onnx4j.opsets.v4.OperatorV4;
import org.onnx4j.tensor.DataType;

/**
 * Concat-4
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat
 * @version This version of the operator has been available since version 4 of
 *          the default ONNX operator set.
 *
 */
public interface ConcatV4<T_TENSOR> extends ConcatV1<T_TENSOR>, OperatorV4 {

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.allTypes();
	}

}
