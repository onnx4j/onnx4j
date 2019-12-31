/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.aiOnnx.v6;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.aiOnnx.v5.AiOnnxOperatorSetSpecV5;
import org.onnx4j.opsets.aiOnnx.v6.ops.CastV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.DropoutV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.MulV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SigmoidV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.aiOnnx.v6.ops.SumV6;

/**
 * Default ONNX Operator Set in version 2
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOperatorSetSpecV6<T_TENSOR> extends AiOnnxOperatorSetSpecV5<T_TENSOR> {
	
	public abstract MulV6<T_TENSOR> getMulV6();
	
	public abstract DropoutV6<T_TENSOR> getDropoutV6();
	
	public abstract CastV6<T_TENSOR> getCastV6();

	public abstract SubV6<T_TENSOR> getSubV6();

	public abstract SumV6<T_TENSOR> getSumV6();

	public abstract SigmoidV6<T_TENSOR> getSigmoidV6();

	//@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOperatorSetSpecV5.super.initializeOperators();
		// 20191115
		operators.put(MulV6.OP_TYPE, this.getMulV6());
		// 20191120
		operators.put(DropoutV6.OP_TYPE, this.getDropoutV6());
		// 20191216
		operators.put(CastV6.OP_TYPE, this.getCastV6());
		// 20191227
		operators.put(SubV6.OP_TYPE, this.getSubV6());
		// 20191230
		operators.put(SumV6.OP_TYPE, this.getSumV6());
		// 20191231
		operators.put(SigmoidV6.OP_TYPE, this.getSigmoidV6());
		return operators;
	}

}