package org.onnx4j.opsets.v5;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v4.OperatorSetSpecV4;
import org.onnx4j.opsets.v5.ops.ReshapeV5;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV5<T_TENSOR> extends OperatorSetSpecV4<T_TENSOR> {

	public abstract ReshapeV5<T_TENSOR> getReshapeV5();

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV4.super.initializeOperators();
		// 20191120
		operators.put(ReshapeV5.OP_TYPE, this.getReshapeV5());
		return operators;
	}

}
