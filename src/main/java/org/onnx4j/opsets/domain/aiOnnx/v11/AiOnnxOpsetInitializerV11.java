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
package org.onnx4j.opsets.domain.aiOnnx.v11;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.domain.aiOnnx.v10.AiOnnxOpsetInitializerV10;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.ReduceMaxV11;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.SoftmaxV11;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.SqueezeV11;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.UnsqueezeV11;

/**
 * Default ONNX Operator Set in version 9
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOpsetInitializerV11 extends AiOnnxOpsetInitializerV10 {

	public abstract SoftmaxV11 getSoftmaxV11();

	public abstract SqueezeV11 getSqueezeV11();

	public abstract UnsqueezeV11 getUnsqueezeV11();

	public abstract ReduceMaxV11 getReduceMaxV11();

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOpsetInitializerV10.super.initializeOperators();
		// 20191231
		operators.put(SoftmaxV11.OP_TYPE, this.getSoftmaxV11());
		// 20200108
		operators.put(SqueezeV11.OP_TYPE, this.getSqueezeV11());
		// 20200110
		operators.put(UnsqueezeV11.OP_TYPE, this.getUnsqueezeV11());
		// 20200113
		operators.put(ReduceMaxV11.OP_TYPE, this.getReduceMaxV11());
		return operators;
	}

}