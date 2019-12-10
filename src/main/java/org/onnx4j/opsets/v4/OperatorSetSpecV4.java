package org.onnx4j.opsets.v4;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v3.OperatorSetSpecV3;
import org.onnx4j.opsets.v4.ops.ConcatV4;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV4<T_TENSOR> extends OperatorSetSpecV3<T_TENSOR> {
	
	public abstract ConcatV4<T_TENSOR> getConcatV4();

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV3.super.initializeOperators();
		// 20191119
		operators.put(ConcatV4.OP_TYPE, this.getConcatV4());
		return operators;
	}

}
