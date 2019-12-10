package org.onnx4j.opsets.v8;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v7.OperatorSetSpecV7;

/**
 * Default ONNX Operator Set in version 8
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV8<T_TENSOR> extends OperatorSetSpecV7<T_TENSOR> {

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV7.super.initializeOperators();
		return operators;
	}

}
