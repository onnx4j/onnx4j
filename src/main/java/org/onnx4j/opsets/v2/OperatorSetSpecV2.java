package org.onnx4j.opsets.v2;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.v1.OperatorSetSpecV1;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface OperatorSetSpecV2<T_TENSOR> extends OperatorSetSpecV1<T_TENSOR> {

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = OperatorSetSpecV1.super.initializeOperators();
		return operators;
	}

}
